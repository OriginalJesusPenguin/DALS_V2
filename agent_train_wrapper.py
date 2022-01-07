import os
import sys
from glob import glob

import train


def main(args):
    # Load info from environment variables
    exp_dir = os.environ['EXP_DIR']
    sweep_id = os.environ['SWEEP_ID']
    exp_postfix = os.environ['EXP_POSTFIX']

    # Deduce current run number and create run directory
    run_number = len(glob(os.path.join(exp_dir, 'run_*'))) + 1

    run_dir = os.path.join(exp_dir, f'run_{run_number}')
    os.mkdir(run_dir)

    # Add arguments
    args += [
        '--checkpoint_dir', run_dir,
        '--checkpoint_postfix', f'{sweep_id}_{exp_postfix}',
        '--experiment_name', f'A{exp_postfix}/R{run_number}',
    ]

    # Start training script
    train.main(args)


if __name__ == '__main__':
    main(sys.argv[1:])

