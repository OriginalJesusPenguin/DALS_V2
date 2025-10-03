import os
import argparse
import trimesh
import math
import datetime
from collections import defaultdict
from time import perf_counter as time

import wandb
from tqdm import tqdm

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import (
    Meshes,
    join_meshes_as_batch,
)

from util import seed_everything
from util.sample import sample_in_ball
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
        parser.add_argument('--weight_grad_loss', type=float, default=5e1)
        parser.add_argument('--weight_zero_set_loss', type=float, default=3e3)
        parser.add_argument('--weight_normal_align_loss', type=float,
                            default=1e2)
        parser.add_argument('--weight_nonzero_set_loss', type=float,
                            default=1e2)
        parser.add_argument('--weight_lv_norm', type=float, default=1e-4)
        parser.add_argument('--num_point_samples', type=int, default=20000)
        parser.add_argument('--learning_rate_net', type=float, default=1e-4)
        parser.add_argument('--learning_rate_lv', type=float, default=1e-4)
        parser.add_argument('--lr_step', type=int, default=500)
        parser.add_argument('--lr_reduce_factor', type=float, default=0.5)

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


    def train(self, train_meshes, val_meshes):
        seed_everything(self.random_seed)
        num_train_samples = len(train_meshes)

        t_samp = time()
        print('Sampling data...')
        sampling_batch_size = 100  # TODO: Maybe make this a param.
        self.train_mesh_point_samples = []
        self.train_mesh_normal_samples = []
        for ib in range(0, num_train_samples, sampling_batch_size):
            ie = min(num_train_samples, ib + sampling_batch_size)
            samples = sample_points_from_meshes(
                join_meshes_as_batch([m
                                      for m in train_meshes[ib:ie]]),
                num_samples=self.num_point_samples // 2,
                return_normals=True,
            )

            # samples = trimesh.sample_
            self.train_mesh_point_samples.append(samples[0].cpu())
            self.train_mesh_normal_samples.append(samples[1].cpu())

        self.train_mesh_point_samples = torch.cat(
            self.train_mesh_point_samples
        )
        self.train_mesh_normal_samples = torch.cat(
            self.train_mesh_normal_samples
        )
        self.train_random_point_samples = sample_in_ball(
            self.num_point_samples // 2
        ).to(self.device)  # Shared by all meshes so just store on device
        t_samp = time() - t_samp 
        print(f'Sampled data in {t_samp:.2f} seconds')

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
        
        # Print available losses with non-zero weights at the beginning
        print(f"\n{'='*70}")
        print("ACTIVE LOSS COMPONENTS:")
        print(f"{'='*70}")
        active_losses = []
        
        # Check each loss component and its weight
        loss_components = [
            ('Grad', self.weight_grad_loss, 'Gradient loss'),
            ('Zero Set', self.weight_zero_set_loss, 'Zero set loss'),
            ('Normal Align', self.weight_normal_align_loss, 'Normal alignment'),
            ('Nonzero Set', self.weight_nonzero_set_loss, 'Nonzero set loss'),
            ('LV Norm', self.weight_lv_norm, 'Latent vector norm')
        ]
        
        for loss_name, weight, description in loss_components:
            if weight != 0:
                active_losses.append((loss_name, weight, description))
                print(f"  {loss_name:<12}: weight = {weight:>8.6f}  ({description})")
        
        if not active_losses:
            print("  WARNING: No active loss components found!")
        else:
            print(f"\n  Total active components: {len(active_losses)}")
        print(f"{'='*70}\n")
        
        try:
            from tqdm import tqdm
            epoch_progress = tqdm(range(self.num_epochs), 
                                desc='Training Progress',
                                ncols=120,
                                initial=0,
                                total=self.num_epochs)
            
            for epoch in epoch_progress:
                epoch_progress.set_description(f'Epoch {epoch + 1}/{self.num_epochs}')
                epoch_profile_times = defaultdict(list)
                epoch_losses = defaultdict(float)
                num_epoch_batches = 0
                t_epoch = time()
                
                # Create progress bar for this epoch
                batch_progress = tqdm(train_index_loader, 
                                    desc=f'Batches',
                                    leave=False,
                                    ncols=100)
                
                for batch_idxs in batch_progress:
                    num_epoch_batches += 1

                    # Prepare batch
                    t_batch = time()
                    batch = self.prepare_batch(batch_idxs)
                    t_batch = time() - t_batch
                    epoch_profile_times['batch'].append(t_batch)

                    # Forward
                    t_forward = time()
                    preds = dict()
                    batch['mesh_points'].requires_grad = True
                    decoder_input_mesh = {
                        'coords': torch.cat(
                            [batch['latent_vectors'], batch['mesh_points']],
                            dim=1
                        )
                    }
                    preds['mesh'] = self.decoder(decoder_input_mesh)
                    batch['random_points'].requires_grad = True
                    decoder_input_rand = {
                        'coords': torch.cat(
                            [batch['latent_vectors'], batch['random_points']],
                            dim=1
                        )
                    }
                    preds['random'] = self.decoder(decoder_input_rand)
                    t_forward = time() - t_forward
                    epoch_profile_times['forward'].append(t_forward)

                    # Loss
                    t_loss = time()
                    losses = self.compute_losses(preds, batch)
                    ramp = min(1.0, epoch / 100.0)
                    
                    # Compute weighted loss components
                    weighted_losses = {}
                    loss = torch.tensor(0.0, device=self.device)
                    
                    # Gradient loss
                    if self.weight_grad_loss != 0:
                        weighted_grad = self.weight_grad_loss * losses['grad']
                        loss += weighted_grad
                        weighted_losses['grad'] = weighted_grad.item()
                    
                    # Zero set loss
                    if self.weight_zero_set_loss != 0:
                        weighted_zero_set = self.weight_zero_set_loss * losses['zero_set']
                        loss += weighted_zero_set
                        weighted_losses['zero_set'] = weighted_zero_set.item()
                    
                    # Normal align loss
                    if self.weight_normal_align_loss != 0:
                        weighted_normal_align = self.weight_normal_align_loss * losses['normal_align']
                        loss += weighted_normal_align
                        weighted_losses['normal_align'] = weighted_normal_align.item()
                    
                    # Nonzero set loss
                    if self.weight_nonzero_set_loss != 0:
                        weighted_nonzero_set = self.weight_nonzero_set_loss * losses['nonzero_set']
                        loss += weighted_nonzero_set
                        weighted_losses['nonzero_set'] = weighted_nonzero_set.item()
                    
                    # LV norm loss (with ramp)
                    if self.weight_lv_norm != 0:
                        weighted_lv_norm = self.weight_lv_norm * ramp * losses['lv_norm']
                        loss += weighted_lv_norm
                        weighted_losses['lv_norm'] = weighted_lv_norm.item()
                    
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
                    epoch_losses['loss'] += loss.item()
                    for name, loss_val in losses.items():
                        epoch_losses[name] += loss_val.item()
                    t_accum = time() - t_accum
                    epoch_profile_times['accum'].append(t_accum)
                    
                    # Update progress bar with weighted loss components
                    postfix_dict = {
                        'Total': f'{loss.item():.4f}',
                    }
                    
                    # Add weighted loss components (only if active)
                    if 'grad' in weighted_losses:
                        postfix_dict['Grad'] = f"{weighted_losses['grad']:.4f}"
                    if 'zero_set' in weighted_losses:
                        postfix_dict['ZeroSet'] = f"{weighted_losses['zero_set']:.4f}"
                    if 'normal_align' in weighted_losses:
                        postfix_dict['NormalAlign'] = f"{weighted_losses['normal_align']:.4f}"
                    if 'nonzero_set' in weighted_losses:
                        postfix_dict['NonzeroSet'] = f"{weighted_losses['nonzero_set']:.4f}"
                    if 'lv_norm' in weighted_losses:
                        postfix_dict['LVNorm'] = f"{weighted_losses['lv_norm']:.4f}"
                    
                    batch_progress.set_postfix(postfix_dict)

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
                        wandb.summary['zero_set'] = epoch_losses['zero_set']
                        wandb.summary['best_epoch'] = self.best_epoch

                    # Save new best model
                    if not self.no_checkpoints:
                        self.save_checkpoint(epoch)
                    t_save = time() - t_save
                    self.profile_times['save'].append(t_save)

                self.scheduler.step()

                t_epoch = time() - t_epoch
                self.profile_times['epoch'].append(t_epoch)

                # Update epoch progress bar with weighted loss components
                epoch_postfix = {
                    'Total': f'{epoch_losses["loss"]:.4f}',
                }
                
                # Add weighted loss components for epoch summary (only if active)
                if self.weight_grad_loss != 0:
                    epoch_postfix['Grad'] = f'{self.weight_grad_loss * epoch_losses["grad"]:.4f}'
                if self.weight_zero_set_loss != 0:
                    epoch_postfix['ZeroSet'] = f'{self.weight_zero_set_loss * epoch_losses["zero_set"]:.4f}'
                if self.weight_normal_align_loss != 0:
                    epoch_postfix['NormalAlign'] = f'{self.weight_normal_align_loss * epoch_losses["normal_align"]:.4f}'
                if self.weight_nonzero_set_loss != 0:
                    epoch_postfix['NonzeroSet'] = f'{self.weight_nonzero_set_loss * epoch_losses["nonzero_set"]:.4f}'
                if self.weight_lv_norm != 0:
                    ramp = min(1.0, epoch / 100.0)
                    epoch_postfix['LVNorm'] = f'{self.weight_lv_norm * ramp * epoch_losses["lv_norm"]:.4f}'
                
                # Add timing info
                epoch_postfix['Time'] = f'{t_epoch:.1f}s'
                
                epoch_progress.set_postfix(epoch_postfix)

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
        mesh_points = self.train_mesh_point_samples[batch_idxs].to(self.device)
        mesh_points = mesh_points.view(-1, 3)
        mesh_normals = self.train_mesh_normal_samples[batch_idxs].to(self.device)
        mesh_normals = mesh_normals.view(-1, 3)
        random_points = self.train_random_point_samples  # Already on device
        random_points = random_points.repeat(len(batch_idxs), 1)

        # Get latent vectors
        points_per_sample = self.num_point_samples // 2
        batch_idxs = batch_idxs.to(self.device)
        batch_idxs = batch_idxs.repeat_interleave(points_per_sample)
        latent_vectors = self.latent_vectors(batch_idxs)

        return {
            'mesh_points': mesh_points,
            'mesh_normals': mesh_normals,
            'random_points': random_points,
            'latent_vectors': latent_vectors,
        }


    def compute_losses(self, preds, batch):
        mesh_sdf = preds['mesh']['model_out']
        mesh_coords = batch['mesh_points']
        rand_sdf = preds['random']['model_out']
        rand_coords = batch['random_points']
        
        grad_mesh = gradient(mesh_sdf, mesh_coords)
        grad_rand = gradient(rand_sdf, rand_coords)
        grad_loss = torch.abs(grad_mesh.norm(dim=-1) - 1).sum() \
                  + torch.abs(grad_rand.norm(dim=-1) - 1).sum()
        grad_loss /= len(grad_mesh) + len(grad_rand)

        zero_set_loss = torch.abs(mesh_sdf).mean()
        normal_align_loss = torch.mean(
            1 - F.cosine_similarity(grad_mesh, batch['mesh_normals'].unsqueeze(0), dim=-1)
        )

        nonzero_set_loss = torch.exp(-1e2 * rand_sdf.abs()).mean()

        lv_norm_loss = torch.mean(torch.norm(batch['latent_vectors'], dim=1))

        return {
            'grad': grad_loss,
            'zero_set': zero_set_loss,
            'normal_align': normal_align_loss,
            'nonzero_set': nonzero_set_loss,
            'lv_norm': lv_norm_loss,
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

        fname = type(self).__name__ + '_'
        fname += self.train_start_time.strftime('%Y-%m-%d_%H-%M')
        if len(self.checkpoint_postfix) > 0:
            fname += f'_{self.checkpoint_postfix}'
        fname += '.ckpt'
        fname = os.path.join(self.checkpoint_dir, fname)
        print('Saving checkpoint:', fname)
        torch.save(state, fname)

