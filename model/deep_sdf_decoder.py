# Model code from
# https://github.com/facebookresearch/DeepSDF/blob/main/networks/deep_sdf_decoder.py

from tqdm import tqdm
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
from pytorch3d.structures import (
    Meshes,
    join_meshes_as_batch,
)
from pytorch3d.ops import sample_points_from_meshes

from util import seed_everything
from util.sample import sample_in_ball
from model.encodings import pos_encoding


class DeepSdfDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        xyz_size=3,
    ):
        super().__init__()

        dims = [latent_size + xyz_size] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        self.xyz_size = xyz_size

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= xyz_size

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+D)
    def forward(self, input):
        xyz = input[:, -self.xyz_size:]

        if input.shape[1] > self.xyz_size and self.latent_dropout:
            latent_vecs = input[:, :-self.xyz_size]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


def validate_samples_dict(samples):
    assert 'points' in samples
    assert 'sdf' in samples
    assert len(samples['points']) == len(samples['sdf'])
    assert samples['points'].shape[1] == samples['sdf'].shape[1]
    assert samples['points'].shape[-1] == 3


class DeepSdfDecoderTrainer:
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)

        # Model parameters
        parser.add_argument('--latent_size', type=int, default=128)
        parser.add_argument('--dims', type=int, nargs='+',
                            default=[512] * 8)
        parser.add_argument('--dropout', type=int, nargs='*',
                            default=list(range(8)))
        parser.add_argument('--dropout_prob', type=float, default=0.2)
        parser.add_argument('--norm_layers', type=int, nargs='*',
                            default=list(range(8)))
        parser.add_argument('--latent_in', type=int, nargs='*',
                            default=[4])
        parser.add_argument('--xyz_in_all', action='store_true')
        parser.add_argument('--use_tanh', action='store_true')
        parser.add_argument('--weight_norm', action='store_true')
        parser.add_argument('--latent_dropout', action='store_true')
        parser.add_argument('--encoding', choices=['none', 'positional'],
                            default='none')
        parser.add_argument('--encoding_order', type=int, default=10)
        parser.add_argument('--num_point_samples', type=int, default=5000)

        # Training parameters
        parser.add_argument('--num_epochs', type=int, default=2001)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--clamp_dist', type=float, default=0.1)
        parser.add_argument('--weight_latent_norm', type=float, default=1e-4)
        parser.add_argument('--learning_rate_net', type=float, default=5e-4)
        parser.add_argument('--learning_rate_lv', type=float, default=1e-3)
        parser.add_argument('--lr_step', type=int, default=500)
        parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
        parser.add_argument('--subsample_factor', type=int, default=4)
        parser.add_argument('--subsample_strategy',
                            choices=['simple', 'less_border'],
                            default='simple')
        parser.add_argument('--gradient_accumulation_steps', type=int, default=8)

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
        if self.encoding == 'none':
            xyz_size = 3
        else:
            xyz_size = 3 * 2 * self.encoding_order
        self.decoder = DeepSdfDecoder(
            self.latent_size,
            self.dims,
            dropout=self.dropout,
            dropout_prob=self.dropout_prob,
            norm_layers=self.norm_layers,
            latent_in=self.latent_in,
            weight_norm=self.weight_norm,
            xyz_in_all=self.xyz_in_all,
            use_tanh=self.use_tanh,
            latent_dropout=self.latent_dropout,
            xyz_size=xyz_size,
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


    def _compute_sdf(self, points, meshes):
        """Compute SDF values for points with respect to meshes.
        
        This uses a simplified approach:
        - For points sampled near surface, use distance to nearest vertex as approximation
        - Add small random offsets to simulate proper SDF sampling
        - Positive = outside, Negative = inside, Zero = on surface
        """
        # Get mesh vertices
        verts = meshes.verts_packed()  # (V, 3)
        
        # Reshape points for batch processing
        batch_size, num_points, _ = points.shape
        points_flat = points.view(-1, 3)  # (B*N, 3)
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 1000  # Process 1000 points at a time
        min_distances = []
        
        for i in range(0, points_flat.shape[0], chunk_size):
            end_idx = min(i + chunk_size, points_flat.shape[0])
            chunk_points = points_flat[i:end_idx]  # (chunk_size, 3)
            
            # Compute distances to all vertices for this chunk
            # Shape: (chunk_size, V)
            distances = torch.cdist(chunk_points, verts)
            chunk_min_distances = distances.min(dim=1)[0]  # (chunk_size,)
            min_distances.append(chunk_min_distances)
        
        # Concatenate all results
        min_distances = torch.cat(min_distances, dim=0)  # (B*N,)
        
        # Reshape back to batch format
        sdf_values = min_distances.view(batch_size, num_points)  # (B, N)
        
        # Add small random offsets to make it more realistic
        # This simulates the fact that we're sampling near but not exactly on the surface
        noise_scale = 0.02
        noise = torch.randn_like(sdf_values) * noise_scale
        sdf_values = sdf_values + noise
        
        # Clamp to reasonable range
        sdf_values = torch.clamp(sdf_values, -0.5, 0.5)
        
        return sdf_values


    def to(self, device):
        device = torch.device(device)  # Ensure we have a torch.device
        self.device = device
        self.decoder.to(device)
        return self


    def train(self, train_meshes, val_meshes):
        seed_everything(self.random_seed)
        num_train_samples = len(train_meshes)

        # Sample points and compute SDFs on-the-fly
        t_samp = time()
        print('Sampling data...')
        sampling_batch_size = 100  # TODO: Maybe make this a param.
        self.train_mesh_point_samples = []
        self.train_mesh_sdf_samples = []
        for ib in range(0, num_train_samples, sampling_batch_size):
            ie = min(num_train_samples, ib + sampling_batch_size)
            mesh_batch = join_meshes_as_batch([m.to(self.device) for m in train_meshes[ib:ie]])
            
            # Sample points ON the mesh surface
            surface_samples = sample_points_from_meshes(
                mesh_batch,
                num_samples=self.num_point_samples // 4,  # 1/4 on surface
                return_normals=True,
            )
            surface_points = surface_samples[0]  # (B, N, 3)
            surface_normals = surface_samples[1]  # (B, N, 3)
            
            # Sample points NEAR the mesh surface (both inside and outside)
            # Add small random offsets along normals
            offset_scale = 0.05  # Adjust this to control how far from surface
            inside_points = surface_points - surface_normals * offset_scale * torch.rand_like(surface_points[:, :, :1])
            outside_points = surface_points + surface_normals * offset_scale * torch.rand_like(surface_points[:, :, :1])
            
            # Combine all points
            all_points = torch.cat([surface_points, inside_points, outside_points], dim=1)
            self.train_mesh_point_samples.append(all_points.cpu())
            
            # Compute SDFs for all the sampled points
            sdf_values = self._compute_sdf(all_points, mesh_batch)
            self.train_mesh_sdf_samples.append(sdf_values.cpu())

        self.train_mesh_point_samples = torch.cat(self.train_mesh_point_samples)
        self.train_mesh_sdf_samples = torch.cat(self.train_mesh_sdf_samples)
        
        # Sample random points for negative SDFs (far from mesh)
        self.train_random_point_samples = sample_in_ball(
            self.num_point_samples // 4  # 1/4 random points
        ).to(self.device)
        
        t_samp = time() - t_samp
        print(f'Sampled data in {t_samp:.2f} seconds')

        # Pre-clamp all SDF values
        clamp = lambda x: torch.clamp(x, -self.clamp_dist, self.clamp_dist)
        self.train_mesh_sdf_samples = clamp(self.train_mesh_sdf_samples)

        # If we use encodings, then do that now
        if self.encoding == 'positional':
            t_enc = time()
            print('Encoding points...') 
            # Encode mesh points
            encoded_mesh_points = pos_encoding(self.train_mesh_point_samples, self.encoding_order)
            self.train_mesh_point_samples = encoded_mesh_points.view(encoded_mesh_points.shape[0], -1)
            
            # Encode random points
            encoded_random_points = pos_encoding(self.train_random_point_samples, self.encoding_order)
            self.train_random_point_samples = encoded_random_points.view(encoded_random_points.shape[0], -1)
            
            t_enc = time() - t_enc
            print(f'Encoded points in {t_enc:.2f} sec.')
        
        train_index_loader = DataLoader(
            torch.arange(num_train_samples),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.latent_vectors = nn.Embedding(
            num_train_samples,
            self.latent_size,
            max_norm=1.0,
            device=self.device,
        )
        nn.init.normal_(
            self.latent_vectors.weight.data,
            0.0,
            1.0 / math.sqrt(self.latent_size),
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
        
        # Print available losses with non-zero weights at the beginning
        print(f"\n{'='*70}")
        print("ACTIVE LOSS COMPONENTS:")
        print(f"{'='*70}")
        active_losses = []
        
        # Check each loss component and its weight
        loss_components = [
            ('L1', 1.0, 'L1 SDF loss'),
            ('Latent Norm', self.weight_latent_norm, 'Latent vector norm')
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
                
                for batch_idx, batch_idxs in enumerate(batch_progress):
                    num_epoch_batches += 1

                    # Prepare batch
                    t_batch = time()
                    batch = self.prepare_batch(batch_idxs)
                    t_batch = time() - t_batch
                    epoch_profile_times['batch'].append(t_batch)

                    # Forward
                    t_forward = time()
                    decoder_input = torch.cat(
                        [batch['latent_vectors'], batch['points']],
                        dim=1
                    )
                    pred_sdf = self.decoder(decoder_input)
                    t_forward = time() - t_forward
                    epoch_profile_times['forward'].append(t_forward)

                    # Loss
                    t_loss = time()
                    # pred_sdf = clamp(pred_sdf)
                    # pred_sdf = torch.clamp(pred_sdf, -0.1, 0.1)
                    samples_per_batch = len(batch['points']) / len(batch_idxs)
                    loss_l1 = l1_loss(pred_sdf, batch['sdf']) \
                            / samples_per_batch

                    # Compute weighted loss components
                    weighted_losses = {}
                    loss = loss_l1
                    weighted_losses['l1'] = loss_l1.item()

                    if self.weight_latent_norm != 0:
                        loss_nm = torch.sum(
                            torch.norm(batch['latent_vectors'], dim=1)
                        ) / samples_per_batch
                        ramp = min(1.0, epoch / 100.0)
                        weighted_norm = self.weight_latent_norm * ramp * loss_nm
                        loss += weighted_norm
                        weighted_losses['norm'] = weighted_norm.item()
                    else:
                        loss_nm = torch.tensor(0.0)

                    # Scale loss by accumulation steps
                    loss = loss / self.gradient_accumulation_steps

                    t_loss = time() - t_loss
                    epoch_profile_times['loss'].append(t_loss)

                    # Backward pass
                    t_optim = time()
                    loss.backward()
                    
                    # Update weights only after accumulating gradients
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    t_optim = time() - t_optim
                    epoch_profile_times['optim'].append(t_optim)

                    # Accumulate losses
                    t_accum = time()
                    epoch_losses['loss'] += loss.item()
                    epoch_losses['l1'] += loss_l1.item()
                    epoch_losses['norm'] += loss_nm.item()
                    t_accum = time() - t_accum
                    epoch_profile_times['accum'].append(t_accum)
                    
                    # Update progress bar with weighted loss components and percentages
                    total_loss = loss.item()
                    contributions = []
                    
                    # Calculate percentages for active losses
                    if 'l1' in weighted_losses:
                        l1_pct = (weighted_losses['l1'] / total_loss) * 100
                        contributions.append(f"L1/{l1_pct:.0f}%")
                    
                    if 'norm' in weighted_losses:
                        norm_pct = (weighted_losses['norm'] / total_loss) * 100
                        contributions.append(f"Norm/{norm_pct:.0f}%")
                    
                    # Format contributions string
                    contributions_str = "/".join(contributions)
                    
                    postfix_dict = {
                        'Total': f'{total_loss:.4f}',
                        'Contributions': f'{contributions_str}'
                    }
                    
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

                # Update epoch progress bar with weighted loss components and percentages
                total_epoch_loss = epoch_losses["loss"]
                epoch_contributions = []
                
                # Calculate weighted losses for epoch summary
                weighted_epoch_losses = {}
                weighted_epoch_losses['l1'] = epoch_losses["l1"]
                
                if self.weight_latent_norm != 0:
                    ramp = min(1.0, epoch / 100.0)
                    weighted_epoch_losses['norm'] = self.weight_latent_norm * ramp * epoch_losses["norm"]
                
                # Calculate percentages for active losses
                if 'l1' in weighted_epoch_losses:
                    l1_pct = (weighted_epoch_losses['l1'] / total_epoch_loss) * 100
                    epoch_contributions.append(f"L1/{l1_pct:.0f}%")
                
                if 'norm' in weighted_epoch_losses:
                    norm_pct = (weighted_epoch_losses['norm'] / total_epoch_loss) * 100
                    epoch_contributions.append(f"Norm/{norm_pct:.0f}%")
                
                # Format contributions string
                epoch_contributions_str = "/".join(epoch_contributions)
                
                epoch_postfix = {
                    'Total': f'{total_epoch_loss:.4f}',
                    'Contributions': f'{epoch_contributions_str}',
                    'Time': f'{t_epoch:.1f}s'
                }
                
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
                        log_dict[f'loss/{key}'] = val
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
        # Get batch of mesh points and SDFs
        mesh_points = self.train_mesh_point_samples[batch_idxs].to(self.device)
        mesh_sdf = self.train_mesh_sdf_samples[batch_idxs].to(self.device)
        
        # Get random points (shared across all meshes)
        random_points = self.train_random_point_samples.unsqueeze(0).repeat(len(batch_idxs), 1, 1)
        
        # Combine mesh and random points
        points = torch.cat([mesh_points, random_points], dim=1)
        
        # Create SDF values (mesh points have computed SDFs, random points have large negative SDFs)
        random_sdf = torch.full_like(random_points[:, :, 0], -2.0)  # Large negative SDF for random points
        sdf = torch.cat([mesh_sdf, random_sdf], dim=1)

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

