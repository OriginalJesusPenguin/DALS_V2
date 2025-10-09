from tqdm import tqdm
import sys
import os
import datetime
import argparse
from time import perf_counter as time
from typing import Sequence
from collections import defaultdict

import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch3d.utils
from pytorch3d.ops import (
    SubdivideMeshes,
    sample_points_from_meshes,
)
from pytorch3d.structures import (
    Meshes,
    join_meshes_as_batch,
)
from pytorch3d.io import save_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
)
from pytorch3d.transforms import random_rotations

from model.graph_conv import MyGraphConv
from model.encodings import sph_encoding, pos_encoding
from model.loss import mesh_bl_quality_loss
from util import seed_everything
try:
    from util.remesh import remesh_template_from_deformed
except ImportError:
    # Remesh functionality not available
    def remesh_template_from_deformed(*args, **kwargs):
        raise NotImplementedError("Remesh functionality not available - missing libremesh.so")


class GraphConvBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None,
                 norm='n'):
        super().__init__()

        assert norm in ['n', 'b', 'l']

        if hidden_features is None:
            hidden_features = []
        features = [in_features] + hidden_features
        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i, o in zip(features[:-1], features[1:]):
            self.graph_convs.append(MyGraphConv(i, o, normalize=True))
            if norm == 'n':
                self.norms.append(nn.Identity())
            elif norm == 'b':
                self.norms.append(nn.BatchNorm1d(o))
            elif norm == 'l':
                self.norms.append(nn.LayerNorm(o))
        self.graph_convs.append(
            MyGraphConv(features[-1], out_features, normalize=True)
        )

        self.activation = nn.ReLU()  # TODO: Make this an input if needed


    def forward(self, vert_features, edges):
        for gc, norm in zip(self.graph_convs[:-1], self.norms):
            vert_features = gc(vert_features, edges)
            vert_features = norm(vert_features)
            vert_features = self.activation(vert_features)

        vert_features = self.graph_convs[-1](vert_features, edges)

        return vert_features


class MeshOffsetBlockGCNN(nn.Module):
    def __init__(self, latent_features, hidden_features=None,
                 vert_in_features=3, vert_out_features=3, norm='n'):
        super().__init__()

        assert norm in ['n', 'b', 'l']

        self.graph_conv_block = GraphConvBlock(
            vert_in_features + latent_features,
            vert_out_features,
            hidden_features,
            norm,
        )


    def forward(self, meshes, latent_vectors, vert_features=None,
                expand_lv=True):
        if vert_features is None:
            vert_features = meshes.verts_packed()
        else:
            vf_dim = vert_features.shape[-1]
            vert_features = vert_features.reshape(-1, vf_dim)

        if expand_lv:
            expanded_lv = latent_vectors[meshes.verts_packed_to_mesh_idx()]
        else:
            expanded_lv = latent_vectors
        vert_features = torch.cat(
            [vert_features, expanded_lv], dim=-1
        )
        edges = meshes.edges_packed()
        offsets = self.graph_conv_block(vert_features, edges)
        return meshes.offset_verts(offsets.view(-1, 3))


class MLP(nn.Sequential):
    def __init__(self, features, norm='n'):
        super().__init__()

        assert norm in ['n', 'b', 'l']

        for i, o in zip(features[:-2], features[1:-1]):
            self.append(nn.Linear(i, o))
            if norm == 'b':
                self.append(nn.BatchNorm1d(o))
            elif norm == 'l':
                self.append(nn.LayerNorm(o))
            self.append(nn.ReLU())
        self.append(nn.Linear(features[-2], features[-1]))


    def append(self, module):
        self.add_module(str(len(self)), module)
        return self


class MeshOffsetBlockMLP(nn.Module):
    """
    MeshOffsetBlockMLP applies a sequence of MLPs to mesh vertex features,
    concatenating latent vectors at each stage as specified by concat_latent_at.
    """
    def __init__(self, latent_features, hidden_features=None,
                 vert_in_features=3, vert_out_features=3, norm='n',
                 concat_latent_at=None):
        super().__init__()

        # Ensure normalization type is valid
        assert norm in ['n', 'b', 'l']

        # If concat_latent_at is not provided, default to empty list
        if concat_latent_at is None:
            concat_latent_at = []
        # feature_runs defines the start and end indices for each MLP block
        feature_runs = [0] + concat_latent_at + [None]

        # If hidden_features is not provided, default to empty list
        if hidden_features is None:
            hidden_features = []
        # features is the list of layer sizes for the full MLP stack
        features = [vert_in_features] + hidden_features + [vert_out_features]

        mlps = []
        # For each block, create an MLP with the appropriate input/output sizes
        for b, e in zip(feature_runs[:-1], feature_runs[1:]):
            if e is not None:
                e += 1  # include the endpoint in the slice
            f = features[b:e]
            # Add latent_features to the input dimension for each block
            f[0] += latent_features
            mlps.append(MLP(f, norm=norm))

        # Store the list of MLPs as a ModuleList for proper registration
        self.mlps = nn.ModuleList(mlps)

    def forward(self, meshes, latent_vectors, vert_features=None,
                expand_lv=True):
        if vert_features is None:
            vert_features = meshes.verts_packed()
        else:
            vf_dim = vert_features.shape[-1]
            vert_features = vert_features.reshape(-1, vf_dim)

        if expand_lv:
            expanded_lv = latent_vectors[meshes.verts_packed_to_mesh_idx()]
        else:
            expanded_lv = latent_vectors

        for mlp in self.mlps:
            vert_features = torch.cat(
                [vert_features, expanded_lv], dim=-1
            )
            vert_features = F.relu(mlp(vert_features))
        return meshes.offset_verts(vert_features.view(-1, 3))


class MeshDecoder(nn.Module):
    def __init__(self, latent_features, steps, hidden_features=None,
                 subdivide=False, mode='gcnn', vert_in_features=3,
                 vert_out_features=3, norm='n', concat_latent_at=None):
        super().__init__()

        if norm not in ['n', 'b', 'l']:
            raise ValueError(
                f"norm must be 'n', 'b', or 'l' for (n)one, (b)atch or (l)ayer normalization but was: {norm}"
            )

        mode = mode.lower()
        if mode == 'gcnn':
            MeshOffsetBlock = MeshOffsetBlockGCNN
            if concat_latent_at is not None and len(concat_latent_at) > 0:
                raise ValueError('concat_latent_at is not supported for GCNN')
        elif mode == 'mlp':
            MeshOffsetBlock = MeshOffsetBlockMLP
        else:
            raise ValueError(f"mode must be 'gcnn' or 'mlp' but was: {mode}")

        if subdivide:
            self.subdivide = SubdivideMeshes()
        else:
            self.subdivide = nn.Identity()
        if mode == 'mlp':
            self.offset_blocks = nn.ModuleList([
                MeshOffsetBlock(
                    latent_features,
                    hidden_features,
                    vert_in_features,
                    vert_out_features,
                    norm,
                    concat_latent_at=concat_latent_at,
                )
                for _ in range(steps)
            ])
        else:  # mode == 'gcnn'
            self.offset_blocks = nn.ModuleList([
                MeshOffsetBlock(
                    latent_features,
                    hidden_features,
                    vert_in_features,
                    vert_out_features,
                    norm,
                )
                for _ in range(steps)
            ])


    def forward(self, templates, latent_vectors, template_vert_features=None,
                expand_lv=True):
        out = []
        pred = templates
        pred = self.offset_blocks[0](templates, latent_vectors,
                                     vert_features=template_vert_features,
                                     expand_lv=expand_lv)
        out.append(pred)
        for block in self.offset_blocks[1:]:
            pred = self.subdivide(pred)
            pred = block(pred, latent_vectors, expand_lv=expand_lv)
            out.append(pred)

        return out


class MeshDecoderTrainer:
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)

        # Model parameters
        parser.add_argument('--latent_features', type=int, default=128)
        parser.add_argument('--steps', type=int, default=1)
        parser.add_argument('--subdivide', type=bool, default=False)
        parser.add_argument('--hidden_features', type=int, nargs='+',
                            default=[256, 256, 128])
        parser.add_argument('--concat_latent_at', type=int, nargs='*',
                            default=[])
        parser.add_argument('--template_subdiv', type=int, default=3)
        parser.add_argument('--decoder_mode', default='gcnn',
                            choices=['gcnn', 'mlp'])
        parser.add_argument('--encoding', default='none',
                            choices=['none', 'spherical_harmonics',
                                     'positional'])
        parser.add_argument('--encoding_order', type=int, default=8)
        parser.add_argument('--normalization', default='none',
                            choices=['none', 'batch', 'layer'])
        parser.add_argument('--rotate_template', action='store_true')
        parser.add_argument('--remesh_every', type=int, default=0)
        parser.add_argument('--remesh_every_edge_length_ratio', type=float,
                            default=1.0)
        parser.add_argument('--remesh_at', type=int, nargs='*', default=[])
        parser.add_argument('--remesh_at_edge_length_ratio', type=float,
                            nargs='*', default=[])

        # Training parameters
        parser.add_argument('--num_epochs', type=int, default=9999)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--weight_normal_loss', type=float, default=0)
        parser.add_argument('--weight_norm_loss', type=float, default=1e-3)
        parser.add_argument('--weight_edge_loss', type=float, default=1e-2)
        parser.add_argument('--weight_laplacian_loss', type=float, default=1e-2)
        parser.add_argument('--weight_quality_loss', type=float, default=0)
        parser.add_argument('--num_mesh_samples', type=int, default=10000)
        parser.add_argument('--train_batch_size', type=int, default=16)
        parser.add_argument('--learning_rate_net', type=float, default=1e-3)
        parser.add_argument('--learning_rate_lv', type=float, default=1e-3)
        parser.add_argument('--lr_reduce_factor', type=float, default=0.1)
        parser.add_argument('--lr_reduce_patience', type=int, default=100)
        parser.add_argument('--lr_reduce_min_lr', type=float, default=1e-5)
        parser.add_argument('--save_shapes_every', type=int, default=0)
        parser.add_argument('--save_shapes_dir', type=str, default='./saved_shapes')

        # Misc. parameters
        parser.add_argument('--no_checkpoints', action='store_true')
        parser.add_argument('--checkpoint_postfix', type=str, default='')
        parser.add_argument('--checkpoint_dir', type=str, default='.')
        parser.add_argument('--random_seed', type=int, default=1337)
        parser.add_argument('--resume_from')

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

        # Extra input validation
        if hparams['rotate_template'] and hparams['encoding'] != 'none':
            raise ValueError('random template rotations are not supported for input encodings')

        if self.remesh_every <= 0:
            self.remesh_every = self.num_epochs + 1

        if len(self.remesh_at_edge_length_ratio) != len(self.remesh_at):
            if len(self.remesh_at_edge_length_ratio) == 0:
                self.remesh_at_edge_length_ratio = [1.0] * len(self.remesh_at)
            else:
                raise ValueError(f'if remesh_at is specified then remesh_at_edge_length_ratio must either be empty or have same length as remesh_at')

        # Initialize model
        template_subdiv = hparams['template_subdiv']
        if hparams['subdivide']:
            template_subdiv -= hparams['steps'] + 1

        self.template = pytorch3d.utils.ico_sphere(template_subdiv)
        # self.template_scale_verts_(0.1)
        self.template.scale_verts_(0.1)
        if hparams['encoding'] == 'spherical_harmonics':
            self.template_encoding = torch.as_tensor(sph_encoding(
                self.template.verts_packed(),
                hparams['encoding_order'],
            ))
            vert_in_feats = (hparams['encoding_order'] + 1) ** 2
        elif hparams['encoding'] == 'positional':
            vert_in_feats = 3 * 2 * hparams['encoding_order']
            self.template_encoding = pos_encoding(
                self.template.verts_packed(),
                hparams['encoding_order'],
            ).view(-1, vert_in_feats)
        else:  # hparams['encoding'] == 'none'
            self.template_encoding = None
            vert_in_feats = 3

        self.decoder = MeshDecoder(
            latent_features=hparams['latent_features'],
            steps=hparams['steps'],
            hidden_features=hparams['hidden_features'],
            concat_latent_at=hparams['concat_latent_at'],
            subdivide=hparams['subdivide'],
            mode=hparams['decoder_mode'],
            vert_in_features=vert_in_feats,
            norm=hparams['normalization'][0],
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
        self.template = self.template.to(device)
        if self.template_encoding is not None:
            self.template_encoding = self.template_encoding.to(device)
        self.latent_vectors = self.latent_vectors.to(device)
        return self


    def train(self, train_meshes, val_meshes):
        seed_everything(self.random_seed)
        num_train_samples = len(train_meshes)

        t_samp = time()
        print('Sampling data...')
        sampling_batch_size = 100  # TODO: Maybe make this a param.
        self.train_point_samples = []
        self.train_normal_samples = []
        for ib in range(0, num_train_samples, sampling_batch_size):
            ie = min(num_train_samples, ib + sampling_batch_size)
            
            # Debug: Check meshes before sampling
            batch_meshes = [m.to(self.device) for m in train_meshes[ib:ie]]
            batch_meshes_cpu = [m.cpu() for m in batch_meshes]
            samples = sample_points_from_meshes(
                join_meshes_as_batch(batch_meshes_cpu),
                num_samples=self.num_mesh_samples,
                return_normals=True,
            )
            # Move samples back to the target device
            samples = [s.to(self.device) for s in samples]
            self.train_point_samples.append(samples[0].cpu())
            self.train_normal_samples.append(samples[1].cpu())

        self.train_point_samples = torch.cat(self.train_point_samples)
        self.train_normal_samples = torch.cat(self.train_normal_samples)
        t_samp = time() - t_samp
        print(f'Sampled data in {t_samp:.2f} seconds')

        train_index_loader = DataLoader(
            torch.arange(num_train_samples),
            batch_size=self.train_batch_size,
            shuffle=True,
        )
        self.latent_vectors = nn.Embedding(
            num_train_samples,
            self.latent_features,
            max_norm=1.0,
            device=self.device,
        )

        # Extending the template is slow so precomute it here
        self.template_extended = self.template.extend(self.batch_size)
        if self.template_encoding is not None:
            self.template_encoding_extended = self.template_encoding.expand(
                self.batch_size, -1, -1
            )
        else:
            self.template_encoding_extended = None

        # Set up optmizer and scheduler
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.lr_reduce_patience,
            factor=self.lr_reduce_factor,
            min_lr=self.lr_reduce_min_lr,
        )

        # Create shapes directory if saving is enabled
        if self.save_shapes_every > 0:
            os.makedirs(self.save_shapes_dir, exist_ok=True)
            self.shapes_saved_count = 0

        # Print available losses with non-zero weights at the beginning
        print(f"\n{'='*70}")
        print("ACTIVE LOSS COMPONENTS:")
        print(f"{'='*70}")
        active_losses = []
        
        # Check each loss component and its weight
        loss_components = [
            ('Chamfer', 1.0, 'Chamfer distance'),
            ('Normal', self.weight_normal_loss, 'Normal consistency'),
            ('Edge Length', self.weight_edge_loss, 'Edge length'),
            ('Laplacian', self.weight_laplacian_loss, 'Laplacian smoothing'),
            ('Quality', self.weight_quality_loss, 'Mesh quality'),
            ('Norm', self.weight_norm_loss, 'Latent vector norm')
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

        # Run training loop
        if self.resume_from is None:
            print('Starting training')
            self.train_start_time = datetime.datetime.now()
            self.profile_times = defaultdict(list)
            self.best_loss = np.inf
            self.best_epoch = -1
            self.best_epoch_losses = defaultdict(float)
            start_epoch = 0
        else:
            print('Resuming training from checkpoint:', self.resume_from)
            self.train_start_time = datetime.datetime.now()
            start_epoch = self.load_checkpoint(self.resume_from)
        try:
            from tqdm import tqdm
            epoch_progress = tqdm(range(start_epoch, self.num_epochs), 
                                desc='Training Progress',
                                ncols=120,
                                initial=start_epoch,
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
                    preds = self.decoder(
                        batch['templates'],
                        batch['latent_vectors'],
                        template_vert_features=batch['template_encodings'],
                    )
                    t_forward = time() - t_forward
                    epoch_profile_times['forward'].append(t_forward)

                    # Loss
                    t_loss = time()
                    losses = self.compute_losses(preds, batch)
                    ramp = min(1.0, epoch / 100.0)
                    
                    # Compute weighted loss components
                    weighted_losses = {}
                    loss = torch.tensor(0.0, device=self.device)
                    
                    # Chamfer loss (always weighted by 1.0)
                    weighted_chamfer = losses['chamfer']
                    loss += weighted_chamfer
                    weighted_losses['chamfer'] = weighted_chamfer.mean().item()
                    
                    # Normal loss
                    if self.weight_normal_loss != 0:
                        weighted_normal = self.weight_normal_loss * losses['normal']
                        loss += weighted_normal
                        weighted_losses['normal'] = weighted_normal.mean().item()
                    
                    # Edge length loss
                    if self.weight_edge_loss != 0:
                        weighted_edge = self.weight_edge_loss * losses['edge_length']
                        loss += weighted_edge
                        weighted_losses['edge_length'] = weighted_edge.mean().item()
                    
                    # Laplacian loss
                    if self.weight_laplacian_loss != 0:
                        weighted_laplacian = self.weight_laplacian_loss * losses['laplacian']
                        loss += weighted_laplacian
                        weighted_losses['laplacian'] = weighted_laplacian.mean().item()
                    
                    # Quality loss
                    if self.weight_quality_loss != 0:
                        weighted_quality = self.weight_quality_loss * losses['quality']
                        loss += weighted_quality
                        weighted_losses['quality'] = weighted_quality.mean().item()
                    
                    # Norm loss (with ramp) - ensure it's a scalar
                    if self.weight_norm_loss != 0:
                        weighted_norm = self.weight_norm_loss * ramp * losses['norm'].mean()
                        loss += weighted_norm
                        weighted_losses['norm'] = weighted_norm.item()

                    loss = loss.mean()
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
                    for name, loss_val in losses.items():
                        epoch_losses[name] += loss_val.mean()
                    t_accum = time() - t_accum
                    epoch_profile_times['accum'].append(t_accum)
                    
                    # Save shapes every N shapes if enabled
                    if self.save_shapes_every > 0:
                        self.shapes_saved_count += len(batch_idxs)
                        if self.shapes_saved_count >= self.save_shapes_every:
                            self._save_shapes(preds, batch, epoch, num_epoch_batches)
                            self.shapes_saved_count = 0
                    
                    # Update progress bar with weighted loss components and percentages
                    total_loss = loss.item()
                    contributions = []
                    
                    # Calculate percentages for active losses
                    if 'chamfer' in weighted_losses:
                        chamfer_pct = (weighted_losses['chamfer'] / total_loss) * 100
                        contributions.append(f"Chamfer/{chamfer_pct:.0f}%")
                    
                    if 'edge_length' in weighted_losses:
                        edge_pct = (weighted_losses['edge_length'] / total_loss) * 100
                        contributions.append(f"Edge/{edge_pct:.0f}%")
                    
                    if 'laplacian' in weighted_losses:
                        laplacian_pct = (weighted_losses['laplacian'] / total_loss) * 100
                        contributions.append(f"Laplacian/{laplacian_pct:.0f}%")
                    
                    if 'normal' in weighted_losses:
                        normal_pct = (weighted_losses['normal'] / total_loss) * 100
                        contributions.append(f"Normal/{normal_pct:.0f}%")
                    
                    if 'quality' in weighted_losses:
                        quality_pct = (weighted_losses['quality'] / total_loss) * 100
                        contributions.append(f"Quality/{quality_pct:.0f}%")
                    
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

                # Remesh template if time
                if (epoch + 1) % self.remesh_every == 0:
                    t_remesh = time()
                    self._remesh_template_from_mean_shape(
                        self.remesh_every_edge_length_ratio
                    )
                    t_remesh = time() - t_remesh
                    self.profile_times['remesh'].append(t_remesh)

                if (epoch + 1) in self.remesh_at:
                    ra_idx = self.remesh_at.index[epoch + 1]
                    ratio = self.remesh_at_edge_length_ratio[ra_idx]
                    t_remesh = time()
                    self._remesh_template_from_mean_shape(ratio)
                    t_remesh = time() - t_remesh
                    self.profile_times['remesh'].append(t_remesh)


                # Close batch progress bar
                batch_progress.close()
                
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
                        wandb.summary['chamfer'] = epoch_losses['chamfer']
                        wandb.summary['best_epoch'] = self.best_epoch

                    # Save new best model
                    if not self.no_checkpoints:
                        self.save_checkpoint(epoch)
                    t_save = time() - t_save
                    self.profile_times['save'].append(t_save)

                self.scheduler.step(epoch_losses['loss'])

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

                # Update epoch progress bar with weighted loss components and percentages
                total_epoch_loss = epoch_losses["loss"]
                epoch_contributions = []
                
                # Calculate weighted losses for epoch summary
                weighted_epoch_losses = {}
                weighted_epoch_losses['chamfer'] = epoch_losses["chamfer"]
                
                if self.weight_normal_loss != 0:
                    weighted_epoch_losses['normal'] = self.weight_normal_loss * epoch_losses["normal"]
                if self.weight_edge_loss != 0:
                    weighted_epoch_losses['edge_length'] = self.weight_edge_loss * epoch_losses["edge_length"]
                if self.weight_laplacian_loss != 0:
                    weighted_epoch_losses['laplacian'] = self.weight_laplacian_loss * epoch_losses["laplacian"]
                if self.weight_quality_loss != 0:
                    weighted_epoch_losses['quality'] = self.weight_quality_loss * epoch_losses["quality"]
                if self.weight_norm_loss != 0:
                    ramp = min(1.0, epoch / 100.0)
                    weighted_epoch_losses['norm'] = self.weight_norm_loss * ramp * epoch_losses["norm"]
                
                # Calculate percentages for active losses
                if 'chamfer' in weighted_epoch_losses:
                    chamfer_pct = (weighted_epoch_losses['chamfer'] / total_epoch_loss) * 100
                    epoch_contributions.append(f"Chamfer/{chamfer_pct:.0f}%")
                
                if 'edge_length' in weighted_epoch_losses:
                    edge_pct = (weighted_epoch_losses['edge_length'] / total_epoch_loss) * 100
                    epoch_contributions.append(f"Edge/{edge_pct:.0f}%")
                
                if 'laplacian' in weighted_epoch_losses:
                    laplacian_pct = (weighted_epoch_losses['laplacian'] / total_epoch_loss) * 100
                    epoch_contributions.append(f"Laplacian/{laplacian_pct:.0f}%")
                
                if 'normal' in weighted_epoch_losses:
                    normal_pct = (weighted_epoch_losses['normal'] / total_epoch_loss) * 100
                    epoch_contributions.append(f"Normal/{normal_pct:.0f}%")
                
                if 'quality' in weighted_epoch_losses:
                    quality_pct = (weighted_epoch_losses['quality'] / total_epoch_loss) * 100
                    epoch_contributions.append(f"Quality/{quality_pct:.0f}%")
                
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
                
                # Print epoch summary with better formatting
                print(f"Epoch {epoch+1:4d} | Loss: {epoch_losses['loss']:.6f} | Time: {t_epoch:.2f}s")
            else:
                print('All epochs done')
                epoch_progress.close()
        except KeyboardInterrupt:
            print('Keyboard interrupt')
            self.num_epochs = epoch + 1
            epoch_progress.close()

        # Print closing summary
        print('Training finished')
        self.print_training_summary()


    def _save_shapes(self, preds, batch, epoch, batch_num):
        """Save target and predicted shapes for visualization"""
        with torch.no_grad():
            # Get the final predicted mesh (last step)
            pred_mesh = preds[-1]
            
            # Create target mesh from sampled points (approximate)
            target_points = batch['target_points']
            target_normals = batch['target_normals']
            
            # Save predicted meshes
            for i in range(len(pred_mesh)):
                pred_verts = pred_mesh.verts_list()[i].cpu()
                pred_faces = pred_mesh.faces_list()[i].cpu()
                
                pred_filename = f"{self.save_shapes_dir}/pred_epoch{epoch}_batch{batch_num}_shape{i}.obj"
                save_obj(pred_filename, pred_verts, pred_faces)
            
            # Save target point clouds as PLY files (since we don't have faces)
            for i in range(len(target_points)):
                target_pts = target_points[i].cpu()
                target_norms = target_normals[i].cpu()
                
                target_filename = f"{self.save_shapes_dir}/target_epoch{epoch}_batch{batch_num}_shape{i}.ply"
                self._save_point_cloud_ply(target_filename, target_pts, target_norms)
            
            print(f"Saved {len(pred_mesh)} shape pairs to {self.save_shapes_dir}")

    def _save_point_cloud_ply(self, filename, points, normals):
        """Save point cloud as PLY file"""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("end_header\n")
            
            for i in range(len(points)):
                pt = points[i]
                norm = normals[i]
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                       f"{norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f}\n")

    def _remesh_template_from_mean_shape(self, edge_length_ratio):
        with torch.no_grad():
            mean_lv = self.latent_vectors.weight.data.detach()
            mean_lv = mean_lv.mean(dim=0).unsqueeze(0)
            mean_mesh = self.decoder(self.template, mean_lv)[-1]
            self.template = remesh_template_from_deformed(
                mean_mesh,
                self.template,
                ratio=edge_length_ratio,
            )
        # Since the template is now denser in some regions to
        # accomodate the deformed geometry, we can't do random
        # rotations anymore
        self.rotate_template = False


    def prepare_batch(self, batch_idxs):
        target_points = self.train_point_samples[batch_idxs].to(self.device)
        target_normals = self.train_normal_samples[batch_idxs].to(self.device)

        batch_idxs = batch_idxs.to(self.device)
        batch_latent_vectors = self.latent_vectors(batch_idxs)

        if len(batch_idxs) != self.batch_size:
            # We have a spill batch so use slow extend here
            templates = self.template.extend(len(batch_idxs))
            if self.template_encoding is not None:
                template_encodings = self.template_encoding.expand(
                    len(batch_idxs), -1, -1
                )
            else:
                template_encodings = None
        else:
            templates = self.template_extended
            template_encodings = self.template_encoding_extended

        if self.rotate_template:
            rot_mats = random_rotations(len(templates), device=templates.device)
            rot_verts = torch.bmm(templates.verts_padded(), rot_mats)
            templates = Meshes(rot_verts, templates.faces_padded())

        return {
            'target_points': target_points,
            'target_normals': target_normals,
            'latent_vectors': batch_latent_vectors,
            'templates': templates,
            'template_encodings': template_encodings,
        }


    def compute_losses(self, preds, batch):
        num_preds = len(preds)
        loss_cf = 0
        loss_nl = 0
        loss_el = 0
        loss_la = 0
        loss_qa = 0
        for pred in preds:
            pred_points, pred_normals = sample_points_from_meshes(
                pred, self.num_mesh_samples, return_normals=True)

            cf_point, cf_normal = chamfer_distance(
                x=pred_points, y=batch['target_points'],
                x_normals=pred_normals, y_normals=batch['target_normals'],
            )
            loss_cf += cf_point
            loss_nl += cf_normal
            loss_el += mesh_edge_loss(pred)
            loss_la += mesh_laplacian_smoothing(pred)
            loss_qa += mesh_bl_quality_loss(pred)

        loss_cf /= num_preds
        loss_nl /= num_preds
        loss_el /= num_preds
        loss_la /= num_preds
        loss_qa /= num_preds

        loss_nm = torch.sum(batch['latent_vectors'] ** 2, dim=1)

        return {
            'chamfer': loss_cf,
            'normal': loss_nl,
            'edge_length': loss_el,
            'laplacian': loss_la,
            'quality': loss_qa,
            'norm': loss_nm,
        }


    def save_checkpoint(self, epoch):
        state = {
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'latent_vectors': self.latent_vectors,
            'template': self.template,
            'hparams': self.hparams,
            'epoch': epoch,
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'best_epoch_losses': self.best_epoch_losses,
            'profile_times': self.profile_times,
        }

        fname = f"MeshDecoderTrainer_"
        fname += self.train_start_time.strftime('%Y-%m-%d_%H-%M')
        if len(self.checkpoint_postfix) > 0:
            fname += f'_{self.checkpoint_postfix}'
        fname += '.ckpt'
        fname = os.path.join(self.checkpoint_dir, fname)
        print('Saving checkpoint:', fname)
        torch.save(state, fname)


    def load_checkpoint(self, fname):
        state = torch.load(fname)
        self.decoder.load_state_dict(state['decoder_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.latent_vectors = state['latent_vectors']
        self.template = state['template']
        self.hparams = state['hparams']
        epoch = state['epoch']
        self.best_epoch = state['best_epoch']
        self.best_loss = state['best_loss']
        self.best_epoch_losses = state['best_epoch_losses']
        self.profile_times = state['profile_times']
        return epoch


    def print_training_summary(self):
        print(f"Best loss at epoch {self.best_epoch + 1} with loss="
              f"{self.best_epoch_losses['loss']}")
        for key, value in self.best_epoch_losses.items():
            print(f'  {key:15}: {value:8.6f}')
        total_time = datetime.timedelta(
            seconds=np.sum(self.profile_times['epoch'])
        )
        mean_time = np.mean(self.profile_times['epoch'])
        print(f'{self.num_epochs} epochs in {total_time} '
              f'({mean_time:.4f} sec/epoch)')
        print('mean profile times:')
        profile_sum = 0
        for key, value in self.profile_times.items():
            if key == 'epoch':
                continue
            profile_sum += np.mean(value)
            print(f'  {key:10}: {np.mean(value):8.6f} sec')
        print('------------------------------')
        print(f'  sum       : {profile_sum:8.6f} sec')

