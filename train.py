import sys
import os
import argparse
from time import perf_counter as time

import wandb

from model.mesh_decoder import MeshDecoderTrainer
from util.data import load_meshes_in_dir


def main(args):
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--experiment_name', default=None)

    parser = MeshDecoderTrainer.add_argparse_args(parser)

    data_parser = parser.add_argument_group('Data')
    data_parser.add_argument('--data_path', type=str)
    data_parser.add_argument('--num_val_samples', type=int, default=5)

    args = parser.parse_args(args)

    # Start logging
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

    # Train network
    trainer = MeshDecoderTrainer(**vars(args))
    trainer.train(train_data, val_data) 


if __name__ == '__main__':
    main(sys.argv[1:])

