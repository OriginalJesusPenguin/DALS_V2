from typing import Optional, Dict

import numpy as np
import torch
from pytorch3d.structures import Meshes

import plotly.graph_objects as go


def plot_wireframe(
    verts: torch.Tensor,
    faces: torch.Tensor,
    fig: Optional[go.Figure] = None,
    fig_kwargs: Optional[Dict] = None,
    **kwargs
) -> go.Figure:
    """
    Plot mesh as wireframe.
    
    Args:
        verts: (N, 3) Tensor with vertex coordinates.
        faces: (M, 3) Tensor with triangle face indices.
        fig: Optional handle to `plotly.graph_objects.Figure`.
        fig_kwargs: Optional dict with keyword arguments for `fig.add_trace`.
        kwargs: Optional keyword arguments for `plotly.graph_objects.Scatter3d`.
        
    Returns:
        fig: Handle to `plotly.graph_objects.Figure`.
    """
    verts = torch.as_tensor(verts)
    faces = torch.as_tensor(faces)
    mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0))
    return plot_meshes_wireframe(mesh, fig=fig, fig_kwargs=fig_kwargs, **kwargs)
    

def plot_meshes_wireframe(
    meshes: Meshes,
    fig: Optional[go.Figure] = None,
    fig_kwargs: Optional[Dict] = None,
    **kwargs
) -> go.Figure:
    """
    Plot `pytorch3d.structures.Meshes` as wireframe.
    
    Args:
        meshes: Meshes to plot.
        fig: Optional handle to `plotly.graph_objects.Figure`.
        fig_kwargs: Optional dict with keyword arguments for `fig.add_trace`.
        kwargs: Optional keyword arguments for `plotly.graph_objects.Scatter3d`.
        
    Returns:
        fig: Handle to `plotly.graph_objects.Figure`.
    """
    verts = meshes.verts_packed()
    edges = meshes.edges_packed()
    
    if fig is None:
        fig = go.Figure()
        
    if fig_kwargs is None:
        fig_kwargs = dict()
        
    if len(kwargs) == 0:
        kwargs = { 
            'marker': {
                'symbol': 'circle',
                'size': 1,
            }
        }

    plot_edges = []    
    for e in edges:
        plot_edges.append(verts[e])
        plot_edges.append(torch.tensor([[np.nan, np.nan, np.nan]]))
    plot_edges = torch.cat(plot_edges, dim=0)
    fig.add_trace(
        go.Scatter3d(x=plot_edges[:, 0], y=plot_edges[:, 1], z=plot_edges[:, 2], **kwargs),
        **fig_kwargs
    )
    return fig
    
    
def plot_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    fig: Optional[go.Figure] = None,
    fig_kwargs: Optional[Dict] = None,
    **kwargs
) -> go.Figure:
    """
    Plot mesh as surface.
    
    Args:
        verts: (N, 3) Tensor with vertex coordinates.
        faces: (M, 3) tensor with triangle face indices.
        fig: Optional handle to `plotly.graph_objects.Figure`.
        fig_kwargs: Optional dict with keyword arguments for `fig.add_trace`.
        kwargs: Optional keyword arguments for `plotly.graph_objects.Scatter3d`.
        
    Returns:
        fig: Handle to `plotly.graph_objects.Figure`. 
    """
    verts = torch.as_tensor(verts)
    faces = torch.as_tensor(faces)
    return plot_meshes(Meshes([verts], [faces]), 
                       fig=fig, fig_kwargs=fig_kwargs, **kwargs)
    
    
def plot_meshes(
    meshes: Meshes,
    fig: Optional[go.Figure] = None,
    fig_kwargs: Optional[Dict] = None,
    **kwargs
) -> go.Figure:
    """
    Plot `pytorch3d.structures.Meshes` as surface.
    
    Args:
        meshes: Meshes to plot.
        fig: Optional handle to `plotly.graph_objects.Figure`.
        fig_kwargs: Optional dict with keyword arguments for `fig.add_trace`.
        kwargs: Optional keyword arguments for `plotly.graph_objects.Scatter3d`.
        
    Returns:
        fig: Handle to `plotly.graph_objects.Figure`.
    """
    if fig is None:
        fig = go.Figure()
        
    if fig_kwargs is None:
        fig_kwargs = dict()
        
    if len(kwargs) == 0:
        kwargs = { 'flatshading': True }
        
    for mesh in meshes:
        vertices = mesh.verts_packed()
        faces = mesh.faces_packed()
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                **kwargs
            ),
            **fig_kwargs
        )
    return fig
