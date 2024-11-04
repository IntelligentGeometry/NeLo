import plotly.graph_objects as go
import numpy as np
import plotly.graph_objs as go
import time
import torch
import trimesh
#from plotly import graph_objs as go


def torch_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array(tensor)
    else:
        return tensor


class Visualizer:
    """
    A class to visualize 3D meshes and point clouds using Plotly.
    """
    def __init__(self):
        self.fig = go.Figure()

    def add_mesh(
        self, vertices, triangles, data=None, opacity=1.0, cmax=None, cmin=None
    ):
        vertices, triangles, data = [
            torch_to_numpy(x) for x in [vertices, triangles, data]
        ]
        # Add traces, one for each slider step
        if data is None:
            self.fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    opacity=opacity,
                    visible=True,  # Mesh is always visible
                    name="",  # Don't show legend for mesh
                    showlegend=False,
                    showscale=False,
                )
            )
        else:
            if cmax is None or cmin is None:
                cmax = data.max()
                cmin = data.min()
            
            self.fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    colorscale="Viridis",
                    intensity=data,
                    intensitymode=(
                        "cell" if data.shape[0] == triangles.shape[0] else "vertex"
                    ),
                    name="",
                    opacity=opacity,
                )
            )
        return self

    def add_points(
        self,
        coords,
        data=None,
        point_size=5,
        showscale=True,
        cmax=None,
        cmin=None,
        opacity=1.0,
    ):
        coords = coords.reshape(-1, 3)
        coords, data = [torch_to_numpy(x) for x in [coords, data]]
        if data is None:
            data = np.ones(len(coords))
        else:
            if data.shape.__len__() == 2:
                data = np.mean(data, axis=1)

        coords = coords.reshape(-1, 3)
        if cmax is None or cmin is None:
            cmax = data.max()
            cmin = data.min()
        print("cmin = ", cmin, "cmax = ", cmax)
        self.fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=data,
                    colorscale="Viridis",  # choose a colorscale
                    #colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']], 
                    opacity=opacity,
                    cmax=cmax,
                    cmin=cmin,
                    colorbar=dict(title="", x=-0.1) if showscale else None,
                ),
                name="",
            )
        )
        return self

    def show(self, grid=True):
        self.fig.update_layout(
            {
                "scene": {
                    "xaxis": {"visible": grid},
                    "yaxis": {"visible": grid},
                    "zaxis": {"visible": grid},
                },
                "scene_aspectmode": "data",
            }
        )
        self.fig.show()
        
    def save_html(self, path="my_plot.html"):
        self.fig.write_html(path)


def normalize_mesh(vertices):
    vertices = vertices - vertices.mean(0)
    vertices = vertices / vertices.abs().max()
    return vertices


def get_triangle_centers(vertices, triangles):
    return vertices[triangles].mean(1)


def get_triangle_normals(vertices, triangles):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    return normals

