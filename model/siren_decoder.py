import os
import argparse
import math
import datetime
from collections import defaultdict
from time import perf_counter as time

import wandb

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from util import seed_everything
from model.encodings import pos_encoding
from model.siren.modules import SingleBVPNet


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad


def validate_samples_dict(samples):
    assert 'points' in samples
    assert 'sdf' in samples
    assert len(samples['points']) == len(samples['sdf'])
    assert samples['points'].shape[1] == samples['sdf'].shape[1]
    assert samples['points'].shape[-1] == 3


class SirenDecoderTrainer:
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)

        # Model parameters
        parser.add_argument('--latent_features', type=int, default=128)
        parser.add_argument('--type', default='sine')
        parser.add_argument('--mode', default='mlp',
                            choices=['mlp', 'nerf', 'rbf'])
        parser.add_argument('--hidden_features', type=int, default=256)
        parser.add_argument('--num_hidden_layers', type=int, default=5)
        # TODO: Add remaining parameters

        # Training parameters
        parser.add_argument('--num_epochs', type=int, default=10001)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--clamp_dist', type=float, default=np.inf)
        parser.add_argument('--weight_latent_norm', type=float, default=1e-4)
        parser.add_argument('--learning_rate_net', type=float, default=1e-4)
        parser.add_argument('--learning_rate_lv', type=float, default=1e-4)
        parser.add_argument('--lr_step', type=int, default=500)
        parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
        parser.add_argument('--subsample_factor', type=int, default=1)
        parser.add_argument('--subsample_strategy',
                            choices=['simple', 'less_border'],
                            default='simple')

        # Misc. parameters
        parser.add_argument('--no_checkpoints', action='store_true')
        parser.add_argument('--checkpoint_postfix', type=str, default='')
        parser.add_argument('--checkpoint_dir', type=str, default='.')
        parser.add_argument('--random_seed', type=int, default=1337)

        return parent_parser


    @classmethod
    def default_hparams(cls):
        # Build an argparser and parse an empty list to get default values
        default_parser = cls.add_argparse_args(
            argparse.ArgumentParser()
        )
        return vars(default_parser.parse_args([]))


    def __init__(self, device=None, log_wandb=True, **kwargs):
        # Register hyperparameters
        hparams = self.default_hparams()
        for key, value in hparams.items():
            # Override default with provided value if present
            hparams[key] = kwargs.get(key, value)
            setattr(self, key, hparams[key])
        self.hparams = hparams

        # Initialize model
        self.decoder = SingleBVPNet(
            type=self.type,
            in_features=self.latent_features + 3,
            mode=self.mode,
            hidden_features=self.hidden_features,
            num_hidden_layers=self.num_hidden_layers,
        )
        self.latent_vectors = torch.tensor([])  # Dummy value

        if device is None:
            # No device was specified so keep everything on default
            self.device = self.latent_vectors.device
        else:
            # Device was specified so transfer everything
            # TODO: Create all things directly on device
            self.to(device)

        self.log_wandb = log_wandb


    def to(self, device):
        device = torch.device(device)  # Ensure we have a torch.device
        self.device = device
        self.decoder.to(device)
        return self


    def train(self, train_samples, val_samples):
        validate_samples_dict(train_samples)
        validate_samples_dict(val_samples)
        seed_everything(self.random_seed)

        num_train_samples = len(train_samples['sdf'])
        self.train_samples = train_samples
        self.val_samples = val_samples

        # Pre-clamp all SDF values
        if np.isfinite(self.clamp_dist):
            clamp = lambda x: torch.clamp(x, -self.clamp_dist, self.clamp_dist)
            for i in range(num_train_samples):
                self.train_samples['sdf'][i] = clamp(
                    self.train_samples['sdf'][i]
                )
        else:
            clamp = lambda x: x

        train_index_loader = DataLoader(
            torch.arange(num_train_samples),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.latent_vectors = nn.Embedding(
            num_train_samples,
            self.latent_features,
            max_norm=1.0,
            device=self.device,
        )
        nn.init.normal_(
            self.latent_vectors.weight.data,
            0.0,
            1.0 / math.sqrt(self.latent_features),
        )

        l1_loss = nn.L1Loss(reduction='sum')

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.decoder.parameters(),
                    "lr": self.learning_rate_net,
                },
                {
                    "params": self.latent_vectors.parameters(),
                    "lr": self.learning_rate_lv,
                },
            ]
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            self.lr_step,
            gamma=self.lr_reduce_factor
        )

        # Run training loop
        print('Starting training')
        self.train_start_time = datetime.datetime.now()
        self.profile_times = defaultdict(list)
        self.best_loss = np.inf
        self.best_epoch = -1
        self.best_epoch_losses = defaultdict(float)
        try:
            for epoch in range(self.num_epochs):
                print(f'Epoch {epoch + 1}')
                epoch_profile_times = defaultdict(list)
                epoch_losses = defaultdict(float)
                num_epoch_batches = 0
                t_epoch = time()
                for batch_idxs in train_index_loader:
                    num_epoch_batches += 1

                    # Prepare batch
                    t_batch = time()
                    batch = self.prepare_batch(batch_idxs)
                    t_batch = time() - t_batch
                    epoch_profile_times['batch'].append(t_batch)

                    # Forward
                    t_forward = time()
                    decoder_input = {
                        'coords': torch.cat(
                            [batch['latent_vectors'], batch['points']],
                            dim=1
                        )
                    }
                    decoder_output = self.decoder(decoder_input)
                    coords = decoder_output['model_in']
                    pred_sdf = decoder_output['model_out']
                    t_forward = time() - t_forward
                    epoch_profile_times['forward'].append(t_forward)

                    # Loss
                    t_loss = time()
                    samples_per_batch = len(batch['points']) / len(batch_idxs)
                    loss_l1 = l1_loss(pred_sdf, batch['sdf']) \
                            / samples_per_batch

                    if self.weight_latent_norm != 0:
                        loss_nm = torch.sum(
                            torch.norm(batch['latent_vectors'], dim=1)
                        ) / samples_per_batch

                    ramp = min(1.0, epoch / 100.0)
                    loss = loss_l1 + self.weight_latent_norm * ramp * loss_nm
                    t_loss = time() - t_loss
                    epoch_profile_times['loss'].append(t_loss)

                    # Update weights
                    t_optim = time()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    t_optim = time() - t_optim
                    epoch_profile_times['optim'].append(t_optim)

                    # Accumulate losses
                    t_accum = time()
                    epoch_losses['loss'] += loss
                    epoch_losses['l1'] += loss_l1
                    epoch_losses['norm'] += loss_nm
                    t_accum = time() - t_accum
                    epoch_profile_times['accum'].append(t_accum)

                # Compute mean epoch losses
                for name in epoch_losses.keys():
                    epoch_losses[name] /= num_epoch_batches

                # Compute epoch profile times
                for key, times in epoch_profile_times.items():
                    self.profile_times[key].append(np.sum(times))

                # Save model if it improved
                if epoch_losses['loss'] <= self.best_loss:
                    t_save = time()
                    self.best_loss = epoch_losses['loss']
                    self.best_epoch_losses = epoch_losses
                    self.best_epoch = epoch

                    # Log
                    if self.log_wandb:
                        wandb.summary['loss'] = self.best_loss
                        wandb.summary['l1'] = epoch_losses['l1']
                        wandb.summary['best_epoch'] = self.best_epoch

                    # Save new best model
                    if not self.no_checkpoints:
                        self.save_checkpoint(epoch)
                    t_save = time() - t_save
                    self.profile_times['save'].append(t_save)

                self.scheduler.step()

                t_epoch = time() - t_epoch
                self.profile_times['epoch'].append(t_epoch)

                # Logging
                if self.log_wandb:
                    opt_param_groups = self.optimizer.state_dict()['param_groups']
                    net_lr = opt_param_groups[0]['lr']
                    lv_lr = opt_param_groups[1]['lr']
                    log_dict = {
                        'optim/decoder_lr': net_lr,
                        'optim/lv_lr': lv_lr,
                        'epoch': epoch,
                    }
                    for key, val in epoch_losses.items():
                        log_dict[f'loss/{key}'] = val.detach().item()
                    for key, val in epoch_profile_times.items():
                        log_dict[f'prof/{key}'] = np.sum(val)
                    log_dict['prof/epoch'] = t_epoch
                    wandb.log(log_dict)

                print(f"Loss: {epoch_losses['loss']:.6g}, "
                      f"{t_epoch:.4f} seconds")
            else:
                print('All epochs done')
        except KeyboardInterrupt:
            print('Keyboard interrupt')   
            self.num_epochs = epoch + 1

        # Print closing summary
        print('Training finished')


    def prepare_batch(self, batch_idxs):
        # Get batch
        points = self.train_samples['points'][batch_idxs]
        sdf = self.train_samples['sdf'][batch_idxs]

        # Subsample if needed
        if self.subsample_factor > 1:
            if self.subsample_strategy == 'simple':
                start = torch.randint(self.subsample_factor, (1,)).item()
                points = points[:, start::self.subsample_factor, :]
                sdf = sdf[:, start::self.subsample_factor]
            # TODO: Don't use hardcoded numbers here.
            else:  # self.subsample_strategy == 'less_border':
                points = torch.cat([
                    points[:, :500000:self.subsample_factor * 2, :],
                    points[:, 500000::self.subsample_factor // 10, :]
                ], dim=1)
                sdf = torch.cat([
                    sdf[:, :500000:self.subsample_factor * 2],
                    sdf[:, 500000::self.subsample_factor // 10]
                ], dim=1)

        points_per_sample = points.shape[1]

        # Reshape and send to device
        points = points.view(-1, points.shape[-1]).to(self.device)
        sdf = sdf.view(-1, 1).to(self.device)

        # Get latent vectors
        batch_idxs = batch_idxs.to(self.device)
        batch_idxs = batch_idxs.repeat_interleave(points_per_sample)
        latent_vectors = self.latent_vectors(batch_idxs)

        return {
            'points': points,
            'sdf': sdf,
            'latent_vectors': latent_vectors,
        }


    def save_checkpoint(self, epoch):
        state = {
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'latent_vectors': self.latent_vectors,
            'hparams': self.hparams,
            'epoch': epoch,
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'best_epoch_losses': self.best_epoch_losses,
            'profile_times': self.profile_times,
        }

        fname = f"DeepSdfDecoderTrainer_"
        fname += self.train_start_time.strftime('%Y-%m-%d_%H-%M')
        if len(self.checkpoint_postfix) > 0:
            fname += f'_{self.checkpoint_postfix}'
        fname += '.ckpt'
        fname = os.path.join(self.checkpoint_dir, fname)
        print('Saving checkpoint:', fname)
        torch.save(state, fname)

