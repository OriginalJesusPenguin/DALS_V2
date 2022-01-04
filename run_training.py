import sys
import os
import argparse
from time import perf_counter as time

import wandb

import torch
import pytorch3d.io

from model.mesh_decoder import MeshDecoderTrainer


def load_meshes_in_dir(path):
    mesh_fnames = sorted(os.listdir(path))
    num_meshes = len(mesh_fnames)

    meshes = []
    io = pytorch3d.io.IO()
    for fname in mesh_fnames:
        meshes.append(io.load_mesh(os.path.join(path, fname), 
                                   include_textures=False))

    return meshes


def main(args):
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--trial_id', type=str, default=0)

    parser = MeshDecoderTrainer.add_argparse_args(parser)

    data_parser = parser.add_argument_group('Data')
    data_parser.add_argument('--data_path', type=str)
    data_parser.add_argument('--num_val_samples', type=int, default=5)

    args = parser.parse_args(args)

    # Start logging
    if args.experiment_id is None:
        args.experiment_id = os.getenv('LSB_JOBID', default='NOID')

    wandb.init(
        project='mesh-decoder',
        entity='patmjen',
        name=f'MeshDecoder_{args.experiment_id}/{args.trial_id}',
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

