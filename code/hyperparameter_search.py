import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize
import numpy as np


from utils import per_error
from dataloader import train_dataloader, val_dataloader


lr_bounds, lambda1_bounds, lambda2_bounds, lambda3_bounds, lambda4_bounds, stepsize_bounds, gamma_bounds = (
    1E-1, 1E-6), (0, 10), (0, 10), (0, 10), (0, 10), (10, 150), (1E-1, 1E-4)
parameter_bounds = torch.tensor([lr_bounds, lambda1_bounds, lambda2_bounds,
                                lambda3_bounds, lambda4_bounds, stepsize_bounds, gamma_bounds])


class FetchBestHyperparameters:
    def __init__(self, model, train_set=train_dataloader,
                 val_set=val_dataloader, eval_metric=per_error, n_iters=20,
                 bounds=parameter_bounds):
        self.model = model
        # save initial state of the model
        self.train_set = train_set
        self.val_set = val_set
        self.eval_metric = eval_metric
        self.bounds = bounds
        self.n_iters = n_iters

    def objective_function(self, params):
        # Load model from initial state
        # Train with given hyperparameters on trainset
        # Validate on valset calculate performance using eval metric
        # return performance

        return -1  # Placeholder

    class GaussianProcess(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=train_x.shape[-1]))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    def acquisition_function(self, x, model, likelihood):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(x))
            return observed_pred.mean - observed_pred.stddev

    def bayesian_optimization(self):
        num_hyperparams = self.bounds.shape[0]
        train_x = torch.rand((10, num_hyperparams))
        for i in range(num_hyperparams):
            train_x[:, i] = train_x[:, i] * \
                (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]

        train_y = torch.tensor([self.objective_function(x) for x in train_x])

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = self.GaussianProcess(train_x, train_y, likelihood)

        for i in range(self.n_iters):
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(50):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            model.eval()
            likelihood.eval()

            def wrapped_acquisition(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                return -self.acquisition_function(x_tensor, model, likelihood).numpy()

            next_point_to_probe = minimize(wrapped_acquisition, x0=np.random.rand(
                num_hyperparams), bounds=self.bounds.numpy()).x
            next_point_to_probe = torch.tensor(
                [next_point_to_probe], dtype=torch.float32)

            next_y = self.objective_function(next_point_to_probe)
            train_x = torch.cat([train_x, next_point_to_probe])
            train_y = torch.cat([train_y, torch.tensor([next_y])])

        return train_x, train_y

    def run(self):
        optimal_x, optimal_y = self.bayesian_optimization()
        print("Optimal Hyperparameters:", optimal_x[-1])
        print("Best Performance:", optimal_y[-1])
        return (optimal_x, optimal_y)

