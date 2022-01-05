import os

import pytorch3d.io


def load_meshes_in_dir(path):
    mesh_fnames = sorted(os.listdir(path))
    num_meshes = len(mesh_fnames)

    meshes = []
    io = pytorch3d.io.IO()
    for fname in mesh_fnames:
        meshes.append(io.load_mesh(os.path.join(path, fname), 
                                   include_textures=False))

    return meshes


