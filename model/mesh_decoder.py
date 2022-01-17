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
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
)

from model.graph_conv import MyGraphConv
from augment.point_wolf import PointWOLF


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
                
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class GraphConvBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None):
        super().__init__()
        
        if hidden_features is None:
            hidden_features = []
        features = [in_features] + hidden_features + [out_features]
        self.graph_convs = nn.ModuleList(
            [MyGraphConv(i, o, normalize=True) for i, o in zip(features[:-1], features[1:])]
        )
        
        self.activation = nn.ReLU()  # TODO: Make this an input if needed
        
        
    def forward(self, vert_features, edges):
        for gc in self.graph_convs[:-1]:
            vert_features = gc(vert_features, edges)
            vert_features = self.activation(vert_features)
            
        vert_features = self.graph_convs[-1](vert_features, edges)
        
        return vert_features
    
    
class MeshOffsetBlock(nn.Module):
    def __init__(self, latent_features, hidden_features=None):
        super().__init__()
        
        self.graph_conv_block = GraphConvBlock(
            3 + latent_features, 3, hidden_features
        )
        
        
    def forward(self, meshes, latent_vectors):
        vert_features = meshes.verts_packed()
        
        expanded_lv = latent_vectors[meshes.verts_packed_to_mesh_idx()]
        vert_features = torch.cat(
            [vert_features, expanded_lv], dim=-1
        )
        edges = meshes.edges_packed()
        offsets = self.graph_conv_block(vert_features, edges)
        return meshes.offset_verts(offsets.view(-1, 3))
    
    
class MeshDecoder(nn.Module):
    def __init__(self, latent_features, steps, hidden_features=None,
                 subdivide=False):
        super().__init__()
        
        if subdivide:
            self.subdivide = SubdivideMeshes()
        else:
            self.subdivide = nn.Identity()
        self.offset_blocks = nn.ModuleList([
            MeshOffsetBlock(latent_features, hidden_features)
            for _ in range(steps)
        ])
        
        
    def forward(self, templates, latent_vectors):
        out = []
        pred = templates
        for i, block in enumerate(self.offset_blocks[:-1]):
            pred = block(pred, latent_vectors)
            out.append(pred)            
            pred = self.subdivide(pred)
        
        out.append(self.offset_blocks[-1](pred, latent_vectors))
        return out

        
class MeshDecoderTrainer:
    def __init__(self, device=None, log_wandb=True, **kwargs):
        # Register hyper-parameters
        hparams = MeshDecoderTrainer.default_hparams()
        for key, value in hparams.items():
            # Override default with provided value if present
            hparams[key] = kwargs.get(key, value)
            setattr(self, key, hparams[key])
        self.hparams = hparams

        # Initialize model
        self.decoder = MeshDecoder(
            hparams['latent_features'],
            hparams['steps'],
            hparams['hidden_features'],
            hparams['subdivide'],
        )
        template_subdiv = hparams['template_subdiv']
        if hparams['subdivide']:
            template_subdiv -= hparams['steps'] + 1
        self.template = pytorch3d.utils.ico_sphere(template_subdiv)
        self.template.scale_verts_(0.1)
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
        self.latent_vectors = self.latent_vectors.to(device)
        return self


    @staticmethod
    def default_hparams():
        # Build an argparser and parse an empty list to get defualt values
        default_parser = MeshDecoderTrainer.add_argparse_args(
            argparse.ArgumentParser()
        )
        return vars(default_parser.parse_args([]))


    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MeshDecoderTrainer")

        # Model parameters
        parser.add_argument('--latent_features', type=int, default=128)
        parser.add_argument('--steps', type=int, default=1)
        parser.add_argument('--subdivide', type=bool, default=False)
        parser.add_argument('--hidden_features', type=int, nargs='+',
                            default=[256, 256, 128])
        parser.add_argument('--template_subdiv', type=int, default=3)

        # Training parameters
        parser.add_argument('--num_epochs', type=int, default=9999)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--weight_normal_loss', type=float, default=0)
        parser.add_argument('--weight_norm_loss', type=float, default=1e-3)
        parser.add_argument('--weight_edge_loss', type=float, default=1e-2)
        parser.add_argument('--weight_laplacian_loss', type=float, default=1e-2)
        parser.add_argument('--num_mesh_samples', type=int, default=10000)
        parser.add_argument('--train_batch_size', type=int, default=256)
        parser.add_argument('--learning_rate_net', type=float, default=1e-3)
        parser.add_argument('--learning_rate_lv', type=float, default=1e-3)
        parser.add_argument('--lr_reduce_factor', type=float, default=0.1)
        parser.add_argument('--lr_reduce_patience', type=int, default=100)
        parser.add_argument('--lr_reduce_min_lr', type=float, default=1e-5)

        # Augmentation parameters
        parser.add_argument('--num_mesh_augment', type=int, default=100)
        parser.add_argument('--pw_num_anchor', type=int, default=4)
        parser.add_argument('--pw_sample_type', type=str, default='fps')
        parser.add_argument('--pw_sigma', type=float, default=0.5)
        parser.add_argument('--pw_r_range', type=float, default=10)
        parser.add_argument('--pw_s_range', type=float, default=2)
        parser.add_argument('--pw_t_range', type=float, default=0.25)

        # Misc. parameters
        parser.add_argument('--no_checkpoints', action='store_true')
        parser.add_argument('--checkpoint_postfix', type=str, default='')
        parser.add_argument('--checkpoint_dir', type=str, default='.')
        parser.add_argument('--random_seed', type=int, default=1337)
        
        return parent_parser

    
    def train(self, train_meshes, val_meshes):
        # Store input
        orig_train_meshes = train_meshes
        seed_everything(self.random_seed)

        # Augment and set up dataloaders 
        t_aug = time()
        print('Augmenting data...')
        train_meshes = self.augment_meshes(train_meshes)
        num_train_samples = len(train_meshes)
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')

        t_samp = time()
        print('Sampling data...')
        # Since sample_points_from_meshes uses a lot of memory we perform the
        # sampling in batches to avoid running out of RAM.
        # TODO: Check if batching is actially faster than processing 1-by-1.
        sampling_batch_size = 100  # TODO: Maybe make this a param.
        self.train_point_samples = []
        self.train_normal_samples = []
        for ib in range(0, num_train_samples, sampling_batch_size):
            ie = min(num_train_samples, ib + sampling_batch_size)
            samples = sample_points_from_meshes(
                join_meshes_as_batch([m.to(self.device)
                                      for m in meshes_aug[ib:ie]]),
                num_samples=self.num_mesh_samples,
                return_normals=True,
            )
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
                    preds = self.decoder(batch['templates'],
                                         batch['latent_vectors'])
                    t_forward = time() - t_forward
                    epoch_profile_times['forward'].append(t_forward)

                    # Loss
                    t_loss = time()
                    losses = self.compute_losses(preds, batch)
                    ramp = min(1.0, epoch / 100.0)
                    loss = losses['chamfer'] \
                         + self.weight_normal_loss * losses['normal'] \
                         + self.weight_edge_loss * losses['edge_length'] \
                         + self.weight_laplacian_loss * losses['laplacian'] \
                         + self.weight_norm_loss * ramp * losses['norm']

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

                print(f"Loss: {epoch_losses['loss']:.6g}, "
                      f"{t_epoch:.4f} seconds")
            else:
                print('All epochs done')
        except KeyboardInterrupt:
            print('Keyboard interrupt')   
            self.num_epochs = epoch + 1

        # Print closing summary
        print('Training finished')
        self.print_training_summary()


    def prepare_batch(self, batch_idxs):
        target_points = self.train_point_samples[batch_idxs].to(self.device)
        target_normals = self.train_normal_samples[batch_idxs].to(self.device)

        batch_idxs = batch_idxs.to(self.device)
        batch_latent_vectors = self.latent_vectors(batch_idxs)

        if len(batch_idxs) != self.batch_size:
            # We have a spill batch so use slow extend here
            templates = self.template.extend(len(batch_idxs))
        else:
            templates = self.template_extended

        return {
            'target_points': target_points,
            'target_normals': target_normals,
            'latent_vectors': batch_latent_vectors,
            'templates': templates,
        }


    def compute_losses(self, preds, batch):
        num_preds = len(preds)
        loss_cf = 0
        loss_nl = 0
        loss_el = 0
        loss_la = 0
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
        
        loss_cf /= num_preds
        loss_nl /= num_preds
        loss_el /= num_preds
        loss_la /= num_preds

        loss_nm = torch.sum(batch['latent_vectors'] ** 2, dim=1)

        return {
            'chamfer': loss_cf,
            'normal': loss_nl,
            'edge_length': loss_el,
            'laplacian': loss_la,
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


    def augment_meshes(self, meshes):
        pw = PointWOLF(
            num_anchor=self.pw_num_anchor,
            sample_type=self.pw_sample_type,
            sigma=self.pw_sigma,
            R_range=self.pw_r_range,
            S_range=self.pw_s_range,
            T_range=self.pw_t_range,
        )

        aug_meshes = []

        for mesh in meshes:
            verts = mesh.verts_packed().numpy()
            faces = mesh.faces_packed()
            for _ in range(self.num_mesh_augment):
                aug_meshes.append(Meshes(
                    verts=[torch.as_tensor(pw(verts)[1])],
                    faces=[faces],
                ))

        return aug_meshes

