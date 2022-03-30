import argparse
from collections import defaultdict
import datetime
from multiprocessing import reduction
from time import perf_counter as time
import os

import numpy as np
import torch
import torch.nn as nn

from monai.networks.nets import UNet  # DEBUG
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
)

import wandb

from util import seed_everything


class ConvStep(nn.Sequential):
    """Convolution step. Conv -> Norm -> Act."""
    def __init__(self, channels_in, channels_out, act=True, norm=True):
        super().__init__()
        self.add_module('Conv', nn.Conv3d(channels_in, channels_out,
                                          kernel_size=3, padding=1))
        if act:
            self.add_module('Act', nn.PReLU())
        if norm:
            self.add_module('Norm', nn.InstanceNorm3d(channels_out))


class ConvBlock(nn.Module):
    def __init__(self, channels, residual=True):
        super().__init__()
        self.residual = residual
        self.block_net = nn.Sequential()
        for i, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:])):
            self.block_net.add_module(f'ConvStep{i}', ConvStep(c_in, c_out))


    def forward(self, x):
        cx = self.block_net(x)
        if self.residual:
            return cx + x
        else:
            return cx


class ConvNet(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_channels,
        num_blocks,
        residual=True
    ):
        super().__init__()
        self.add_module('InConv', nn.Conv3d(in_channels, block_channels[0],
                                            kernel_size=1, padding=1))
        for i in range(num_blocks):
            self.add_module(f'Block{i}', ConvBlock(block_channels, residual))
        self.add_module('OutConv', nn.Conv3d(block_channels[-1], out_channels,
                                             kernel_size=1, padding=1))


class ConvNetTrainer:
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)

        # Model parameters
        parser.add_argument('--block_channels', type=int, nargs='+',
                            default=[16, 16])
        parser.add_argument('--num_blocks', type=int, default=1)

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

        # Initialize model
        #self.model = ConvNet(
        #    in_channels=1,
        #    out_channels=1,
        #    block_channels=self.block_channels,
        #    num_blocks=self.num_blocks,
        #).to(device)
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
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
            ToDeviced(keys=all_train_keys, device=self.device),
        ])
        val_transforms = Compose([
            AddChanneld(keys=all_val_keys),
            EnsureTyped(keys=all_val_keys),
            ToDeviced(keys=all_val_keys, device=self.device),
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

        post_pred = Compose([EnsureType()])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        scaler = torch.cuda.amp.GradScaler()

        self.best_metric = -np.inf
        self.best_epoch = -1
        self.train_start_time = datetime.datetime.now()
        t_train = time()
        try:
            for epoch in range(self.num_epochs):
                print(f"Epoch: {epoch + 1}")
                losses = defaultdict(float)
                profile_times = defaultdict(float)
                num_epoch_batches = 0
                t_epoch = time()
                self.model.train()
                for batch in train_loader:
                    num_epoch_batches += 1

                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
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
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    profile_times['optim'] += time() - t_optim

                    losses['loss'] += loss.item()

                for key in losses.keys():
                    losses[key] /= num_epoch_batches

                for key in profile_times.keys():
                    profile_times[key] /= num_epoch_batches

                profile_times['epoch'] = time() - t_epoch

                if epoch % self.eval_every == 0:
                    t_val = time()
                    self.model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            image, label = batch['images'], batch['labels']
                            with torch.cuda.amp.autocast():
                                pred = torch.sigmoid(self.model(image)) > 0.5

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
                    wandb.log(log_dict)

                print(f"Loss: {losses['loss']:.6g}, "
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
