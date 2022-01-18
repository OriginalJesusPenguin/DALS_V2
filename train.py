import sys
import os
import argparse
from time import perf_counter as time

import wandb

from model.mesh_decoder import MeshDecoderTrainer
from util.data import load_meshes_in_dir
from util import seed_everything
from augment.point_wolf import augment_meshes


def main(args):
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--no_wandb', action='store_true')

    parser = MeshDecoderTrainer.add_argparse_args(parser)

    data_parser = parser.add_argument_group('Data')
    data_parser.add_argument('--data_path', type=str)
    data_parser.add_argument('--num_val_samples', type=int, default=5)
    data_parser.add_argument('--num_mesh_augment', type=int, default=100)
    data_parser.add_argument('--data_random_seed', type=int, default=1337)
    data_parser.add_argument('--pw_num_anchor', type=int, default=4)
    data_parser.add_argument('--pw_sample_type', type=str, default='fps')
    data_parser.add_argument('--pw_sigma', type=float, default=0.5)
    data_parser.add_argument('--pw_r_range', type=float, default=10)
    data_parser.add_argument('--pw_s_range', type=float, default=2)
    data_parser.add_argument('--pw_t_range', type=float, default=0.25)

    args = parser.parse_args(args)

    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='mesh-decoder',
            entity='patmjen',
            name=args.experiment_name,
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
    if args.num_mesh_augment > 0:
        t_aug = time()
        print('Augmenting data...')
        seed_everything(args.data_random_seed)
        train_data = augment_meshes(
            train_data,
            num_mesh_augment=args.num_mesh_augment,
            num_anchor=args.pw_num_anchor,
            sample_type=args.pw_sample_type,
            sigma=args.pw_sigma,
            R_range=args.pw_r_range,
            S_range=args.pw_s_range,
            T_range=args.pw_t_range,
        )
        t_aug = time() - t_aug
        print(f'Augmented data in {t_aug:.2f} seconds')

    # Train network
    trainer = MeshDecoderTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data) 


if __name__ == '__main__':
    main(sys.argv[1:])

