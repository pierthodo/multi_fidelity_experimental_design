from math import sqrt
import numpy as np
import plotly.graph_objects as go


def plot(X, Y, model_variance):
    mesh_size = 50
    x_1_space = np.linspace(X[:, 0].min(), X[:, 0].max(), mesh_size)
    x_2_space = np.linspace(X[:, 1].min(), X[:, 1].max(), mesh_size)
    xx, yy = np.meshgrid(x_1_space, x_2_space)
    X_mesh = np.hstack([xx.reshape((-1, 1)), yy.reshape((-1, 1))])
    utility = model_variance.evaluate(X_mesh).reshape((mesh_size, mesh_size))
    fig = go.Figure(
        data=[
            go.Surface(
                x=xx, y=yy, z=utility, opacity=0.7, showscale=False, colorscale="RdBu"
            ),
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=[utility.min()] * X.shape[0],
                mode="markers",
                marker_symbol="x",
                marker_size=5,
            ),
        ]
    )

    dic_scene = dict(
        xaxis_title="Step size", yaxis_title="Diffusivity", zaxis_title="Utility"
    )

    fig.update_layout(
        scene=dic_scene,
    )  # title="Partial melt: "+str(melt_partial)+" lengthscale: "+str(model_gpy.param_array[2]),

    fig.show()
