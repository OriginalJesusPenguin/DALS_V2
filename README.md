# Deep Active Latent Surfaces for Medical Geometries
This is the reference implementation of the method described in

<ul><b>Deep Active Latent Surfaces for Medical Geometries</b>,<br>
    Anonymous authors, Submitted to NeurIPS, 2022.<br>
</ul>

## Installation
First, install:
* PyTorch: https://pytorch.org/
* PyTorch3d: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

This project was developed with PyTorch 1.10.2 and PyTorch3d 0.6.1. 
Make sure you use compatible CUDA versions.

Remaining dependencies can be installed with pip via `pip install -r requirements.txt`.

## Training
First, store your training meshes in some folder, e.g., `~/path/to/meshes`, as `.obj`, `.ply`, or `.off` files.

Then, to train a model with the parameters of the paper, run
```
python train.py mesh_decoder --data_path="~/path/to/meshes"
```
To hold the last `N` meshes our for later validation, add `--num_val_samples=$N`.
The full list of configuration options is:
```
$ python train.py --help

$ python train.py mesh_decoder --help
```

## Dataset
We use the 'Liver Tumours' and 'Spleen' datasets from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/).