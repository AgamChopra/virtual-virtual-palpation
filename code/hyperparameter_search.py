import torch
from torch import nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize
import numpy as np
from tqdm import trange, tqdm


from utils import per_error, ssim_loss, PSNR, getPositionEncoding, compose
from dataloader import train_dataloader, val_dataloader


lr_bounds, lambda1_bounds, lambda2_bounds, lambda3_bounds,\
    lambda4_bounds, stepsize_bounds, gamma_bounds = (1E-1, 1E-6), (0, 10),
(0, 10), (0, 10), (0, 10), (10, 150), (1E-1, 1E-4)
parameter_bounds = torch.tensor([lr_bounds, lambda1_bounds, lambda2_bounds,
                                lambda3_bounds, lambda4_bounds,
                                stepsize_bounds, gamma_bounds])


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


class FetchBestHyperparameters:
    def __init__(self, model, train_set=train_dataloader,
                 val_set=val_dataloader, eval_metric=per_error, n_iters=20,
                 bounds=parameter_bounds, state_name='model', device='cuda',
                 criterion=[nn.MSELoss(), nn.L1Loss(),
                            ssim_loss(win_size=3, win_sigma=0.1), PSNR()],
                 HYAK=True, epochs=1000):
        self.model = model.to(device)
        self.train_set = train_set
        self.val_set = val_set
        self.eval_metric = eval_metric
        self.bounds = bounds
        self.n_iters = n_iters
        self.state_name = state_name
        torch.save(self.model.state_dict(),
                   f'initial_state_{self.state_name}.pt')
        self.device = device
        self.criterion = criterion
        self.HYAK = HYAK
        self.GaussianProcess = GaussianProcess
        self.epochs = epochs

    def objective_function(self, lr, lam1, lam2, lam3, lam4, stepsize, gamma):
        self.model.load_state_dict(torch.load(
            f'initial_state_{self.state_name}.pt', map_location=self.device))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr.item())
        self.lambdas = [lam1.item(), lam2.item(), lam3.item(), lam4.item()]
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=stepsize.item(), gamma=gamma.item())

        for _ in trange(self.epochs, desc='Training'):
            self.step()

        avg_percent_error = self.validate()
        performance = -avg_percent_error
        return performance

    def step(self):
        self.model.train()
        self.optimizer.zero_grad()

        x = self.train_set.load_batch()
        input_signal = x[0].detach().type(torch.float).to(self.device)
        real_output_signal = x[1].detach().type(torch.float).to(self.device)

        if self.state_name == 'taunet':
            initial_masking_signal = input_signal[:, 0:1]
            step_output_i, step_output_j, step_emb_i = self.mask_func.apply(
                real_output_signal, initial_masking_signal)
            masked_input_signal_i = input_signal * \
                torch.where(step_output_i > 0, 1., 0.)
            fake_step_output_j = self.model(masked_input_signal_i, step_emb_i)
            signal = (step_output_j, fake_step_output_j)

        else:
            fake_output_signal = self.model(input_signal)
            signal = (real_output_signal, fake_output_signal)

        error = sum([self.lambdas[i] * self.criterion[i](signal[0], signal[1])
                    for i in range(len(self.criterion))])

        error.backward()
        self.optimizer.step()

    @torch.no_grad()
    def validate(self, TAU=0.1):
        errors = []
        self.model.eval()
        data = self.val_set
        if self.state_name == 'taunet':
            t_enc = getPositionEncoding(
                self.filter_steps-1, 64).to(next(self.model.parameters()
                                                 ).device)

        for x_batch in tqdm(data, desc="Validating"):
            input_signal, real_output_signal = [
                x.to(self.device) for x in x_batch]

            if self.model_type == 'taunet':
                mask = [torch.where(
                    input_signal[:, 0:1] > TAU, 1, 0).cpu()]

                for i in range(self.filter_steps - 1):
                    t = t_enc[i:i+1]
                    t = torch.cat(
                        [t for _ in range(real_output_signal.shape[0])],
                        dim=0)
                    input_signal *= mask[-1].cuda()
                    mask.append(self.model(
                        input_signal, t).cpu() * mask[-1])

                mask = torch.cat(mask, dim=1)
                fake_output_signal = compose(mask)

            else:
                fake_output_signal = self.model(input_signal)

            error = per_error(real_output_signal, fake_output_signal)

            errors.extend(error.cpu())

        avg_err = torch.mean(errors)

        return avg_err

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

        train_y = torch.tensor([self.objective_function(*x) for x in train_x])

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
                return -self.acquisition_function(x_tensor, model,
                                                  likelihood).numpy()

            next_point_to_probe = minimize(
                wrapped_acquisition, x0=np.random.rand(
                    num_hyperparams), bounds=self.bounds.numpy()).x
            next_point_to_probe = torch.tensor(
                [next_point_to_probe], dtype=torch.float32)

            next_y = self.objective_function(next_point_to_probe)
            train_x = torch.cat([train_x, next_point_to_probe])
            train_y = torch.cat([train_y, torch.tensor([next_y])])

        return train_x, train_y

    def search(self):
        print('Bayesian Hyperparameter Search Initiated...')
        optimal_x, optimal_y = self.bayesian_optimization()
        print(f'Optimal Hyperparameters: {optimal_x[-1]}')
        print(f'Best Performance: {optimal_y[-1]}')
        return (optimal_x, optimal_y)
