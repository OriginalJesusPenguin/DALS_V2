import torch
import torch.nn as nn
from pytorch3d import _C
from pytorch3d.ops.graph_conv import gather_scatter, gather_scatter_python


def vert_degrees(verts, edges, directed=False):
    """
    Compute vertex degrees (number of neighbors) for graph.
    
    Args:
        verts: (N, D) Tensor with vertex features.
        edges: (M, 2) Tensor with edges as [from, to] indices.
        directed: Bool indicating if edges in the graph are directed.
        
    Returns:
        degrees: (N,) Tensor with vertex degrees.
    """
    assert verts.device == edges.device, \
        'verts and edges must be on the same device'
    assert verts.dim() == 2, \
        f'verts must be 2 dimensional but was {verts.dim()}'
    assert edges.dim() == 2, \
        f'edges must be 2 dimensional but was {edges.dim()}'
    assert edges.shape[1] == 2, \
        f'edges must have shape (M, 2), but was (M, {edges.shape[1]})'
    
    device = verts.device
    
    idx0, idx1 = edges.unbind(dim=1)
    # Use expand to avoid allocating extra memory.
    ones = torch.ones(1, device=device).expand(edges.shape[0])
    
    degrees = torch.zeros(verts.shape[0], device=device)
    degrees.scatter_add_(0, idx0, ones)
    if not directed:
        degrees.scatter_add_(0, idx1, ones)
    
    return degrees


class MyGraphConv(nn.Module):
    """A single graph convolution layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init: str = "normal",
        directed: bool = False,
        normalize: bool = False,
    ) -> None:
        """
        Args:
            input_dim: Number of input features per vertex.
            output_dim: Number of output features per vertex.
            init: Weight initialization method. Can be one of ['zero', 'normal'].
            directed: Bool indicating if edges in the graph are directed.
            normalize: Bool indicating whether to normalize neighbor sum by vertex degree.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.directed = directed
        self.normalize = normalize
        self.w0 = nn.Linear(input_dim, output_dim)
        self.w1 = nn.Linear(input_dim, output_dim)

        if init == "normal":
            nn.init.normal_(self.w0.weight, mean=0, std=0.01)
            nn.init.normal_(self.w1.weight, mean=0, std=0.01)
            # pyre-fixme[16]: Optional type has no attribute `data`.
            self.w0.bias.data.zero_()
            self.w1.bias.data.zero_()
        elif init == "zero":
            self.w0.weight.data.zero_()
            self.w1.weight.data.zero_()
        else:
            raise ValueError('Invalid GraphConv initialization "%s"' % init)

            
    def forward(self, verts, edges):
        """
        Args:
            verts: FloatTensor of shape (V, input_dim) where V is the number of
                vertices and input_dim is the number of input features
                per vertex. input_dim has to match the input_dim specified
                in __init__.
            edges: LongTensor of shape (E, 2) where E is the number of edges
                where each edge has the indices of the two vertices which
                form the edge.

        Returns:
            out: FloatTensor of shape (V, output_dim) where output_dim is the
            number of output features per vertex.
        """
        if verts.is_cuda != edges.is_cuda:
            raise ValueError("verts and edges tensors must be on the same device.")
        if verts.shape[0] == 0:
            # empty graph.
            return verts.new_zeros((0, self.output_dim)) * verts.sum()

        verts_w0 = self.w0(verts)  # (V, output_dim)
        verts_w1 = self.w1(verts)  # (V, output_dim)

        if torch.cuda.is_available() and verts.is_cuda and edges.is_cuda:
            neighbor_sums = gather_scatter(verts_w1, edges, self.directed)
        else:
            neighbor_sums = gather_scatter_python(
                verts_w1, edges, self.directed
            )  # (V, output_dim)
            
        if self.normalize:
            with torch.no_grad():
                deg = vert_degrees(verts, edges, self.directed).unsqueeze(1)
            neighbor_sums = neighbor_sums / deg

        # Add neighbor features to each vertex's features.
        out = verts_w0 + neighbor_sums
        return out