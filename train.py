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
    # Verify CUDA before doing anything
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Requested device: {device}")
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available but --device cuda was requested!")
            print("Available devices:", torch.cuda.device_count())
            print("Exiting...")
            sys.exit(1)
        
        # Get the GPU that SLURM allocated
        import os
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            cuda_visible = os.environ['CUDA_VISIBLE_DEVICES']
            print(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
        
        # Force CUDA initialization
        try:
            test_tensor = torch.zeros(1).cuda()
            print(f"✓ CUDA initialized successfully")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ Total GPUs: {torch.cuda.device_count()}")
        except RuntimeError as e:
            print(f"ERROR: CUDA initialization failed: {e}")
            print("Exiting...")
            sys.exit(1)
    else:
        print(f"Using device: {args.device}")
    
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='mesh-decoder',
            name='mesh-decoder',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    train_data = load_meshes_in_dir(args.train_data_path)
    val_data = load_meshes_in_dir(args.val_data_path)
    
    # Get training filenames for saving in checkpoint
    import glob
    import os
    train_filenames = sorted(glob.glob(os.path.join(args.train_data_path, '*.obj')))
    train_filenames = [os.path.basename(fname) for fname in train_filenames]
    print(f'Found {len(train_filenames)} training filenames')
    
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
    trainer = MeshDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    # Store data paths and filenames for checkpoint saving
    trainer.train_data_path = args.train_data_path
    trainer.val_data_path = args.val_data_path
    trainer.train_filenames = train_filenames
    trainer.train(train_data, val_data)


def train_local_mesh_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='local-mesh-decoder',
            name='local-mesh-decoder',
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
            name='deep-sdf',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    train_data = load_meshes_in_dir(args.train_data_path)
    val_data = load_meshes_in_dir(args.val_data_path)
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
    trainer = DeepSdfDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data)


def train_siren_decoder(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='siren',
            name='siren',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    train_data = load_meshes_in_dir(args.train_data_path)
    val_data = load_meshes_in_dir(args.val_data_path)
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
            name='siren2',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    train_data = load_meshes_in_dir(args.train_data_path)
    val_data = load_meshes_in_dir(args.val_data_path)
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
            name='local-mod-siren',
            config=args,
            dir=args.checkpoint_dir,
        )

    # Load data
    t_load = time()
    print('Loading data...')
    train_data = load_meshes_in_dir(args.train_data_path)
    val_data = load_meshes_in_dir(args.val_data_path)
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
    parser.add_argument('--no_wandb', action='store_true')

    data_parser = parser.add_argument_group('Data')
    data_parser.add_argument('--train_data_path', type=str)
    data_parser.add_argument('--val_data_path', type=str)
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

