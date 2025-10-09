import argparse
import sys
from time import perf_counter as time

import torch

import wandb

from model.dals_segmenter import ConvNetTrainer


def dict_unzip(d):
    """Turn dict of arrays into list of dicts."""
    num_elems = len(d[list(d.keys())[0]])
    out = [dict() for _ in range(num_elems)]
    for key, array in d.items():
        assert len(array) == num_elems, "All arrays must have same length"
        for i, v in enumerate(array):
            out[i][key] = v
    return out


def simulate_slice_annot(labels, slicing='random'):
    assert slicing in ['center', 'random']

    masks = torch.zeros_like(labels)
    for i, label in enumerate(labels):
        xyz = torch.nonzero(label).float()
        
        # Skip if no non-zero elements (empty label)
        if xyz.shape[0] == 0:
            # If no labels, just pick center slices
            x_slices = label.shape[0] // 2
            y_slices = label.shape[1] // 2
            z_slices = label.shape[2] // 2
        else:
            if slicing == 'center':
                com = xyz.mean(dim=0).round().long()
                x_slices = com[0]
                y_slices = com[1]
                z_slices = com[2]
            elif slicing == 'random':
                bbox_min = xyz.min(dim=0).values
                bbox_max = xyz.max(dim=0).values
                bbox_len = bbox_max - bbox_min
                all_slices = bbox_min + torch.rand(1) * bbox_len
                x_slices = all_slices[0].round().long()
                y_slices = all_slices[1].round().long()
                z_slices = all_slices[2].round().long()

        masks[i, x_slices, :, :] = 1
        masks[i, :, y_slices, :] = 1
        masks[i, :, :, z_slices] = 1

    return masks


def train_conv_net(args):
    # Start logging
    if not args.no_wandb:
        wandb.init(
            project='dals-conv-net',
            # entity='patmjen',
            config=args,
            dir=args.checkpoint_dir,
        )

    t_load = time()
    train_data = torch.load(args.train_data_path)
    if args.simulate_slice_annot:
        train_data['masks'] = simulate_slice_annot(train_data['labels'],
                                                   args.slicing_mode)
    val_data = torch.load(args.val_data_path)

    train_data = dict_unzip(train_data)
    val_data = dict_unzip(val_data)

    train_data = train_data[args.train_subset[0]:args.train_subset[1]]
    val_data = val_data[args.val_subset[0]:args.val_subset[1]]

    t_load = time() - t_load
    print(f'Loaded data in {t_load:.2f} seconds')

    trainer = ConvNetTrainer(log_wandb=(not args.no_wandb), **vars(args))
    trainer.train(train_data, val_data, slice_annot=args.simulate_slice_annot)


def main(argv):
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--no_wandb', action='store_true')

    data_parser = parser.add_argument_group('Data')
    data_parser.add_argument('--train_data_path', type=str)
    data_parser.add_argument('--val_data_path', type=str)
    data_parser.add_argument('--train_subset', type=int, nargs=2,
                             default=[0, -1])
    data_parser.add_argument('--val_subset', type=int, nargs=2,
                             default=[0, -1])
    data_parser.add_argument('--simulate_slice_annot', action='store_true')
    data_parser.add_argument('--slicing_mode', choices=['center', 'random'],
                             default='random')

    subparsers = parser.add_subparsers(dest='segmentation_model')

    conv_net_parser = subparsers.add_parser('conv_net')
    ConvNetTrainer.add_argparse_args(conv_net_parser)

    args = parser.parse_args(argv)
    for k, v in vars(args).items():
        print(k, ':', v)

    if args.segmentation_model == 'conv_net':
        train_conv_net(args)
    else:
        raise ValueError('Invalid segmentation model')


if __name__ == '__main__':
    main(sys.argv[1:])