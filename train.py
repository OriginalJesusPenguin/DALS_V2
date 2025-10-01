import sys
import argparse
from time import perf_counter as time

import torch
import wandb

from model.mesh_decoder import MeshDecoderTrainer
from model.local_mesh_decoder import LocalMeshDecoderTrainer
from model.deep_sdf_decoder import DeepSdfDecoderTrainer
from model.siren_decoder import SirenDecoderTrainer
from model.siren_decoder_2 import SirenDecoderTrainer as SirenDecoderTrainer2
from model.local_mod_siren import LocalModSirenDecoderTrainer
from util.data import load_meshes_in_dir, load_npz_in_dir, split_dict_data
from util import seed_everything
from augment.point_wolf import augment_meshes, augment_points


def train_mesh_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='mesh-decoder',
            name='DALS',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    data = load_meshes_in_dir(args.data_path)
    train_data = data[:-args.num_val_samples]
    val_data = data[-args.num_val_samples:]
    t_load = time() - t_load
    print(f'Loaded data in {t_load:.2f} seconds')

    # Debug: Check for NaN values before augmentation
    print("DEBUG: Checking for NaN values before augmentation...")
    num_nan_verts = 0
    num_inf_verts = 0
    num_nan_faces = 0
    num_inf_faces = 0
    for i, mesh in enumerate(train_data):
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        has_nan_verts = torch.isnan(verts).any()
        has_inf_verts = torch.isinf(verts).any()
        has_nan_faces = torch.isnan(faces.float()).any()
        has_inf_faces = torch.isinf(faces.float()).any()
        if has_nan_verts:
            num_nan_verts += 1
        if has_inf_verts:
            num_inf_verts += 1
        if has_nan_faces:
            num_nan_faces += 1
        if has_inf_faces:
            num_inf_faces += 1
    print(f"DEBUG: Found {num_nan_verts} NaN verts, {num_inf_verts} Inf verts, {num_nan_faces} NaN faces, {num_inf_faces} Inf faces")

    # Augment
    if args.num_augment > 0:
        t_aug = time()
        print('Augmenting data...')
        seed_everything(args.data_random_seed)
        train_data = augment_meshes(
            train_data,
            num_augment=args.num_augment,
            num_anchor=args.pw_num_anchor,
            sample_type=args.pw_sample_type,
            sigma=args.pw_sigma,
            R_range=args.pw_r_range,
            S_range=args.pw_s_range,
            T_range=args.pw_t_range,
            device=args.device,  # Will be forced to CPU in augment_meshes  # Will be forced to CPU in augment_meshes
        )
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')
        
        # Debug: Check for NaN values after augmentation
        print("DEBUG: Checking for NaN values after augmentation...")
        for i, mesh in enumerate(train_data):
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()
            has_nan_verts = torch.isnan(verts).any()
            has_inf_verts = torch.isinf(verts).any()
            has_nan_faces = torch.isnan(faces.float()).any()
            has_inf_faces = torch.isinf(faces.float()).any()
            # print(f"  Mesh {i}: verts NaN={has_nan_verts}, verts Inf={has_inf_verts}, faces NaN={has_nan_faces}, faces Inf={has_inf_faces}")
            # if has_nan_verts or has_inf_verts or has_nan_faces or has_inf_faces:
            #     print(f"  Mesh {i} has problematic values after augmentation!")
            #     print(f"    Verts shape: {verts.shape}, min: {verts.min()}, max: {verts.max()}")
            #     print(f"    Faces shape: {faces.shape}, min: {faces.min()}, max: {faces.max()}")

    # Train network
    trainer = MeshDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data)


def train_local_mesh_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='local-mesh-decoder',
            name=args.experiment_name or 'local-mesh-decoder',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    data = load_meshes_in_dir(args.data_path)
    train_data = data[:-args.num_val_samples]
    val_data = data[-args.num_val_samples:]
    t_load = time() - t_load
    print(f'Loaded data in {t_load:.2f} seconds')

    # Augment
    if args.num_augment > 0:
        t_aug = time()
        print('Augmenting data...')
        seed_everything(args.data_random_seed)
        train_data = augment_meshes(
            train_data,
            num_augment=args.num_augment,
            num_anchor=args.pw_num_anchor,
            sample_type=args.pw_sample_type,
            sigma=args.pw_sigma,
            R_range=args.pw_r_range,
            S_range=args.pw_s_range,
            T_range=args.pw_t_range,
            device=args.device,  # Will be forced to CPU in augment_meshes
        )
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')

    # Train network
    trainer = LocalMeshDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data)


def train_deep_sdf_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='deep-sdf',
            name=args.experiment_name or 'deep-sdf',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    data = load_npz_in_dir(args.data_path, keys=['points', 'sdf'])
    train_data, val_data = split_dict_data(data, -args.num_val_samples)
    t_load = time() - t_load
    print(f'Loaded data in {t_load:.2f} seconds')

    # Augment
    if args.num_augment > 0:
        t_aug = time()
        print('Augmenting data...')
        seed_everything(args.data_random_seed)
        train_data['points'] = augment_points(
            train_data['points'],
            num_augment=args.num_augment,
            num_anchor=args.pw_num_anchor,
            sample_type=args.pw_sample_type,
            sigma=args.pw_sigma,
            R_range=args.pw_r_range,
            S_range=args.pw_s_range,
            T_range=args.pw_t_range,
            device=args.device,  # Will be forced to CPU in augment_meshes
        )
        train_data['sdf'] = train_data['sdf'].repeat_interleave(
            args.num_augment,
            dim=0,
        )
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')

    # Train network
    trainer = DeepSdfDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data)


def train_siren_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='siren',
            name=args.experiment_name or 'siren',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    data = load_meshes_in_dir(args.data_path)
    train_data = data[:-args.num_val_samples]
    val_data = data[-args.num_val_samples:]
    t_load = time() - t_load
    print(f'Loaded data in {t_load:.2f} seconds')

    # Augment
    if args.num_augment > 0:
        t_aug = time()
        print('Augmenting data...')
        seed_everything(args.data_random_seed)
        train_data = augment_meshes(
            train_data,
            num_augment=args.num_augment,
            num_anchor=args.pw_num_anchor,
            sample_type=args.pw_sample_type,
            sigma=args.pw_sigma,
            R_range=args.pw_r_range,
            S_range=args.pw_s_range,
            T_range=args.pw_t_range,
            device=args.device,  # Will be forced to CPU in augment_meshes
        )
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')

    # Train network
    trainer = SirenDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data)


def train_siren2_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='siren',
            name=args.experiment_name or 'siren2',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    data = load_meshes_in_dir(args.data_path)
    train_data = data[:-args.num_val_samples]
    val_data = data[-args.num_val_samples:]
    t_load = time() - t_load
    print(f'Loaded data in {t_load:.2f} seconds')

    # Augment
    if args.num_augment > 0:
        t_aug = time()
        print('Augmenting data...')
        seed_everything(args.data_random_seed)
        train_data = augment_meshes(
            train_data,
            num_augment=args.num_augment,
            num_anchor=args.pw_num_anchor,
            sample_type=args.pw_sample_type,
            sigma=args.pw_sigma,
            R_range=args.pw_r_range,
            S_range=args.pw_s_range,
            T_range=args.pw_t_range,
            device=args.device,  # Will be forced to CPU in augment_meshes  # Will be forced to CPU in augment_meshes
        )
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')

    # Train network
    trainer = SirenDecoderTrainer2(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data)


def train_local_mod_siren_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='siren',
            name=args.experiment_name or 'local-mod-siren',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    data = load_meshes_in_dir(args.data_path)
    train_data = data[:-args.num_val_samples]
    val_data = data[-args.num_val_samples:]
    t_load = time() - t_load
    print(f'Loaded data in {t_load:.2f} seconds')

    # Augment
    if args.num_augment > 0:
        t_aug = time()
        print('Augmenting data...')
        seed_everything(args.data_random_seed)
        train_data = augment_meshes(
            train_data,
            num_augment=args.num_augment,
            num_anchor=args.pw_num_anchor,
            sample_type=args.pw_sample_type,
            sigma=args.pw_sigma,
            R_range=args.pw_r_range,
            S_range=args.pw_s_range,
            T_range=args.pw_t_range,
            device=args.device,  # Will be forced to CPU in augment_meshes
        )
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')

    # Train network
    trainer = LocalModSirenDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data)


def main(args):
    parser = argparse.ArgumentParser()

    # General argument
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--no_wandb', action='store_true')

    data_parser = parser.add_argument_group('Data')
    data_parser.add_argument('--data_path', type=str)
    data_parser.add_argument('--num_val_samples', type=int, default=5)
    data_parser.add_argument('--num_augment', type=int, default=100)
    data_parser.add_argument('--data_random_seed', type=int, default=1337)
    data_parser.add_argument('--pw_num_anchor', type=int, default=4)
    data_parser.add_argument('--pw_sample_type', type=str, default='fps')
    data_parser.add_argument('--pw_sigma', type=float, default=0.5)
    data_parser.add_argument('--pw_r_range', type=float, default=10)
    data_parser.add_argument('--pw_s_range', type=float, default=2)
    data_parser.add_argument('--pw_t_range', type=float, default=0.25)

    # Sub-parsers for each model
    subparsers = parser.add_subparsers(dest='decoder_model')

    mesh_decoder_parser = subparsers.add_parser('mesh_decoder')
    MeshDecoderTrainer.add_argparse_args(mesh_decoder_parser)

    local_mesh_decoder_parser = subparsers.add_parser('local_mesh_decoder')
    LocalMeshDecoderTrainer.add_argparse_args(local_mesh_decoder_parser)

    deep_sdf_parser = subparsers.add_parser('deep_sdf')
    DeepSdfDecoderTrainer.add_argparse_args(deep_sdf_parser)

    siren_parser = subparsers.add_parser('siren')
    SirenDecoderTrainer.add_argparse_args(siren_parser)

    siren2_parser = subparsers.add_parser('siren2')
    SirenDecoderTrainer2.add_argparse_args(siren2_parser)

    local_mod_siren_parser = subparsers.add_parser('local_mod_siren')
    LocalModSirenDecoderTrainer.add_argparse_args(local_mod_siren_parser)

    # Parse and start training
    args = parser.parse_args(args)
    if args.decoder_model == 'mesh_decoder':
        train_mesh_decoder(args)
    elif args.decoder_model == 'local_mesh_decoder':
        train_local_mesh_decoder(args)
    elif args.decoder_model == 'deep_sdf':
        train_deep_sdf_decoder(args)
    elif args.decoder_model == 'siren':
        train_siren_decoder(args)
    elif args.decoder_model == 'siren2':
        train_siren2_decoder(args)
    elif args.decoder_model == 'local_mod_siren':
        train_local_mod_siren_decoder(args)


if __name__ == '__main__':
    main(sys.argv[1:])

