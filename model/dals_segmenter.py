from tqdm import tqdm  # Add tqdm import if not already present
import argparse
from collections import defaultdict
import datetime
from multiprocessing import reduction
from time import perf_counter as time
import os

import numpy as np
import torch
import torch.nn as nn

from monai.networks.nets import (
    UNet,
    VNet,
    DynUNet,
    UNETR,
)
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    decollate_batch,
)
from monai.losses import DiceCELoss, DiceLoss, MaskedDiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    ToDeviced,
    EnsureTyped,
    EnsureType,
    KeepLargestConnectedComponent,
)

import wandb

from util import seed_everything


def compute_dice_score(pred, target, threshold=0.5):
    """
    Compute Dice score for binary segmentation.
    
    Args:
        pred: Predicted probabilities [B, C, H, W, D] or [B, C, H, W]
        target: Ground truth labels [B, C, H, W, D] or [B, C, H, W]
        threshold: Threshold for converting probabilities to binary predictions
    
    Returns:
        Average dice score across batch
    """
    # Convert probabilities to binary predictions
    pred_binary = (torch.softmax(pred, dim=1)[:, 1] > threshold).float()
    
    # Handle target - if it has 2 channels, use the second one, otherwise use the first
    if target.shape[1] == 2:
        target_binary = target[:, 1].float()
    else:
        target_binary = target[:, 0].float()
    
    # Compute intersection and union
    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))
    
    # Compute dice score for each sample in batch
    dice_scores = (2.0 * intersection) / (union + 1e-8)
    
    # Return average dice score across batch
    return dice_scores.mean().item()


def _get_kernels_strides(sizes, spacings):
    # From MONAI DynUNet examaple
    # https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_pipeline/create_network.py
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


class ConvNetTrainer:
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)

        # Model parameters
        parser.add_argument('--model', default='ResUNet',
                            choices=['ResUNet', 'VNet', 'DynUNet', 'UNETR'])
        # TODO: Find better to take sizes as arguments that checks the input
        parser.add_argument('--data_size', type=int, nargs='+',
                            default=[256, 256, 256])
        parser.add_argument('--data_spacing', type=float, nargs='+',
                            default=[1.0, 1.0, 1.0])

        # Training parameters
        parser.add_argument('--num_epochs', type=int, default=99999)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--eval_every', type=int, default=10)

        # Misc. parameters
        parser.add_argument('--no_checkpoints', action='store_true')
        parser.add_argument('--checkpoint_postfix', type=str, default='')
        parser.add_argument('--checkpoint_dir', type=str, default='.')
        parser.add_argument('--random_seed', type=int, default=1337)

        return parent_parser


    @classmethod
    def default_hparams(cls):
        # Build an argparser and parse an empty list to get defualt values
        default_parser = cls.add_argparse_args(
            argparse.ArgumentParser()
        )
        return vars(default_parser.parse_args([]))


    def __init__(self, device=None, log_wandb=True, **kwargs):
        # Register hyper-parameters
        hparams = self.default_hparams()
        for key, value in hparams.items():
            # Override default with provided value if present
            hparams[key] = kwargs.get(key, value)
            setattr(self, key, hparams[key])
        self.hparams = hparams
        self.log_wandb = log_wandb

        if self.model == 'ResUNet':
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        elif self.model == 'VNet':
            self.model = VNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
            )
        elif self.model == 'DynUNet':
            # TODO: Get data size as input
            kernels, strides = _get_kernels_strides(
                self.data_size,
                self.data_spacing,
            )
            self.model = DynUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                kernel_size=kernels,
                strides=strides,
                upsample_kernel_size=strides[1:],
                norm_name='instance',
                # TODO: Do deep supervision
            )
        elif self.model == 'UNETR':
            # From MONAI UNETR tutorial:
            # https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d.ipynb
            self.model = UNETR(
                in_channels=1,
                out_channels=2,
                img_size=self.data_size,
                feature_size=16,
                hidden_size=768//2,
                mlp_dim=3072//2,
                num_heads=12//2,
                pos_embed="conv",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            )
        if device is None:
            device = torch.device('cpu')
        else:
            self.to(device)


    def to(self, device):
        device = torch.device(device)
        self.device = device
        self.model = self.model.to(device)
        return self


    def train(self, train_data, val_data, slice_annot=False):
        seed_everything(self.random_seed)

        # Set up dataloaders
        all_train_keys = ['images', 'labels']
        all_val_keys = ['images', 'labels']
        if slice_annot:
            all_train_keys.append('masks')
        train_transforms = Compose([
            AddChanneld(keys=all_train_keys),
            EnsureTyped(keys=all_train_keys),
            #ToDeviced(keys=all_train_keys, device=self.device),
        ])
        val_transforms = Compose([
            AddChanneld(keys=all_val_keys),
            EnsureTyped(keys=all_val_keys),
            #ToDeviced(keys=all_val_keys, device=self.device),
        ])

        train_ds = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=8,
            copy_cache=False,
        )
        val_ds = CacheDataset(
            data=val_data,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=5,
            copy_cache=False,
        )

        train_loader = ThreadDataLoader(
            train_ds,
            num_workers=0,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = ThreadDataLoader(
            val_ds,
            num_workers=0,
            batch_size=self.batch_size,
        )
        if slice_annot:
            loss_function = MaskedDiceLoss(
                softmax=True,
                squared_pred=True,
                batch=True,
                to_onehot_y=True,
            )
        else:
            loss_function = DiceLoss(
                softmax=True,
                squared_pred=True,
                batch=True,
                to_onehot_y=True,
            )

        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([
            EnsureType(),
            AsDiscrete(to_onehot=2),
            KeepLargestConnectedComponent(1)
        ])
        dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        self.best_metric = -np.inf
        self.best_epoch = -1
        self.train_start_time = datetime.datetime.now()
        t_train = time()
        try:
            for epoch in range(self.num_epochs):
                print(f"Epoch: {epoch + 1}")
                losses = defaultdict(float)
                profile_times = defaultdict(float)
                dice_scores = []

                num_epoch_batches = 0
                t_epoch = time()
                self.model.train()
                for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                    num_epoch_batches += 1
                    # Transfer to device
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)

                    self.optimizer.zero_grad()
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            t_forward = time()
                            pred = self.model(batch['images'])
                            profile_times['forward'] += time() - t_forward
                    else:
                        t_forward = time()
                        pred = self.model(batch['images'])
                        profile_times['forward'] += time() - t_forward

                    t_loss = time()
                    if slice_annot:
                        mask = batch['masks']
                        loss = loss_function(
                            pred * mask,
                            batch['labels'] * mask,
                            mask
                        )
                    else:
                        loss = loss_function(pred, batch['labels'])
                    profile_times['loss'] += time() - t_loss

                    t_optim = time()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    profile_times['optim'] += time() - t_optim

                    losses['loss'] += loss.item()
                    
                    # Compute dice score for this batch
                    with torch.no_grad():
                        dice_score = compute_dice_score(pred, batch['labels'])
                        dice_scores.append(dice_score)

                for key in losses.keys():
                    losses[key] /= num_epoch_batches
                
                # Compute average dice score for this epoch
                avg_dice = np.mean(dice_scores) if dice_scores else 0.0

                for key in profile_times.keys():
                    profile_times[key] /= num_epoch_batches

                profile_times['epoch'] = time() - t_epoch

                if epoch % self.eval_every == 0:
                    t_val = time()
                    self.model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            # Transfer to device
                            for key in batch.keys():
                                if isinstance(batch[key], torch.Tensor):
                                    batch[key] = batch[key].to(self.device)
                            image, label = batch['images'], batch['labels']
                            with torch.cuda.amp.autocast():
                                pred = self.model(image)

                            pred = [post_pred(i)
                                    for i in decollate_batch(pred)]
                            label = [post_label(i)
                                     for i in decollate_batch(label)]
                            dice_metric(y_pred=pred, y=label)

                        metric = dice_metric.aggregate().item()
                        dice_metric.reset()
                        if metric > self.best_metric:
                            self.best_metric = metric
                            self.best_epoch = epoch

                            if self.log_wandb:
                                wandb.summary['dice'] = metric
                                wandb.summary['best_epoch'] = self.best_epoch

                            if not self.no_checkpoints:
                                self.save_checkpoint(epoch)

                    profile_times['validation'] = time() - t_val
                    print(f"Validation loss: {metric:.6g}, "
                          f"{profile_times['validation']:.4f} seconds")


                if self.log_wandb:
                    log_dict = dict()
                    for key, val in losses.items():
                        log_dict[f'Loss/{key}'] = val
                    for key, val in profile_times.items():
                        log_dict[f'Prof/{key}'] = val
                    log_dict['Dice/Train'] = avg_dice
                    wandb.log(log_dict)

                print(f"Loss: {losses['loss']:.6g}, "
                      f"Dice: {avg_dice:.4f}, "
                      f"{profile_times['epoch']:.4f} seconds")

        except KeyboardInterrupt:
            print('Keyboard interrupt')
            self.num_epochs = epoch + 1

        t_train = time() - t_train

        print('Training finished')
        print(f'{self.num_epochs} epochs in {t_train:.2f} seconds.')
        print(f'Best validation loss: {self.best_metric} '
              f'at epoch {self.best_epoch}')


    def save_checkpoint(self, epoch):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hparams': self.hparams,
            'epoch': epoch,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
        }

        fname = type(self).__name__ + '_'
        fname += self.train_start_time.strftime('%Y-%m-%d_%H-%M')
        if len(self.checkpoint_postfix) > 0:
            fname += f'_{self.checkpoint_postfix}'
        fname += '.ckpt'
        fname = os.path.join(self.checkpoint_dir, fname)
        print('Saving checkpoint:', fname)
        torch.save(state, fname)