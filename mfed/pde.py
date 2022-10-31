from math import sqrt
import pde
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
import numpy as np
from pde import PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid
from pde import DiffusionPDE, ScalarField, UnitGrid
from emukit.core.acquisition.acquisition_per_cost import acquisition_per_expected_cost
from tqdm import tqdm 

class pde_runner:
    def __init__(self, param, f_y) -> None:
        self.param = param
        self.update_PDE()
        self.f_y = f_y
        self.cost = []

    def update_PDE(self):
        if self.param["pde_name"] == "diffusion":
            grid = UnitGrid([64, 64])  # generate grid
            self.state = ScalarField.random_uniform(
                grid, 0.2, 0.3, rng=np.random.default_rng(seed=self.param["seed"])
            )  # generate initial condition
            self.eq = DiffusionPDE(diffusivity=self.param["diff"])  # define the pde
            # state.plot(title='Initial state')
        elif self.param["pde_name"] == "brusselator":
            a, b = 1, 3  # 1, 3
            d0, d1 = self.param["diff"], 0.1
            self.eq = PDE(
                {
                    "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                    "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
                }
            )
            # initialize state
            grid = UnitGrid([64, 64])
            u = ScalarField(grid, a, label="Field $u$")
            v = b / a + 0.1 * ScalarField.random_normal(
                grid, label="Field $v$", rng=np.random.default_rng(seed=self.param["seed"])
            )
            self.state = FieldCollection([u, v])

    def __call__(self, X):
        return self.run(X)

    def run(self, X):
        Y = []
        for i in tqdm(range(X.shape[0])):
            self.param["dt"] = X[i, 0]
            self.param["diff"] = X[i, 1]
            self.update_PDE()
            solver = pde.ExplicitSolver(self.eq, adaptive=False)
            solver.dt_max = self.param["dt"]
            solver.dt_min = self.param["dt"]
            storage = MemoryStorage()
            controller1 = pde.Controller(
                solver, t_range=self.param["t_range"], tracker=[storage.tracker(0.02)]
            )  # 20
            sol = controller1.run(self.state, dt=self.param["dt"])
            if (
                len(sol.data.shape) > 2
            ):  # If there are more than one dimension in the PDE select the given one according to param
                sol = sol.data[self.param["dim_data"], :, :].flatten()
            else:
                sol = sol.data.flatten()
            Y.append(self.f_y(sol))
            self.cost.append(
                [
                    X[i, 0],
                    float(controller1.diagnostics["controller"]["profiler"]["solver"]),
                ]
            )
        return np.array(Y).reshape((-1, 1))
