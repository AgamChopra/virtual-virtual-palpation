"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
    - PyTorch 2.0 stable documentation @ https://pytorch.org/docs/stable/
"""
import os
from sys import exit
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from models import get_models
from utils import show_images, per_error, ssim_loss, PSNR, mask_generator,\
    compose, getPositionEncoding


class Trainer(nn.Module):
    def __init__(self, checkpoint_path, dataloader, CH_IN=5, CH_OUT=1, n=1,
                 optimizer=torch.optim.AdamW, learning_rate=1E-4,
                 criterion=[nn.MSELoss(), nn.L1Loss(),
                            ssim_loss(win_size=3, win_sigma=0.1), PSNR()],
                 lambdas=[0.15, 0.15, 0.6, 0.2], device='cuda',
                 model_type='unet', step_size=350, gamma=0.1,
                 filter_steps=256):
        '''
        TO DO...

        Parameters
        ----------
        checkpoint_path : TYPE
            DESCRIPTION.
        dataloader : tuple
            must return a tuple of tensors (batch of input, batch of
                                            ground truth).
        CH_IN : TYPE, optional
            DESCRIPTION. The default is 5.
        CH_OUT : TYPE, optional
            DESCRIPTION. The default is 1.
        n : TYPE, optional
            DESCRIPTION. The default is 1.
        optimizer : TYPE, optional
            DESCRIPTION. The default is torch.optim.AdamW.
        learning_rate : TYPE, optional
            DESCRIPTION. The default is 1E-4.
        criterion : TYPE, optional
            DESCRIPTION. The default is [nn.MSELoss(), nn.L1Loss(),
                                         ssim_loss(win_size=3, win_sigma=0.1),
                                         PSNR()].
        lambdas : TYPE, optional
            DESCRIPTION. The default is [0.15, 0.15, 0.6, 0.2].
        device : TYPE, optional
            DESCRIPTION. The default is 'cuda'.
        model_type : TYPE, optional
            DESCRIPTION. The default is 'unet'.
        step_size : TYPE, optional
            DESCRIPTION. The default is 350.
        gamma : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        '''
        super(Trainer, self).__init__()
        self.model_type = model_type
        models = get_models(CH_IN, CH_OUT, n)
        model = models.get(
            self.model_type) if self.model_type in models else exit(
                f'{self.model_type} not implemented!')
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = torch.compile(model).to(device)
        try:
            print('Loading parameters from last run')
            self.model.load_state_dict(torch.load(os.path.join(
                checkpoint_path, (f'{self.model_type}_autosave.pt'))))
        except Exception:
            try:
                print(
                    'paramerts failed to load from last run\n\
                        Loading saved initial state')
                self.model.load_state_dict(torch.load(
                    f'initial_state_{self.model_type}.pt'))
            except Exception:
                print('paramerts failed to load\nUsing random state')
        self.data = dataloader
        self.iterations = (int((dataloader.max_id + 1) / dataloader.batch) +
                           ((dataloader.max_id + 1) % dataloader.batch > 0))
        self.optimizer = optimizer(self.model.parameters(), learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)
        self.criterion = criterion
        self.lambdas = lambdas
        self.train_error = []
        self.val_error = []
        if self.model_type == 'taunet':
            self.mask_func = mask_generator(filter_steps, 64)

    def step(self):
        self.model.train()
        self.optimizer.zero_grad()

        x = self.data.load_batch()
        input_signal = x[0].detach().type(torch.float).to(self.device)
        real_output_signal = x[1].detach().type(torch.float).to(self.device)

        if self.model_type == 'taunet':
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

        return error.item()

    def optimize(self, epochs=200, HYAK=True, val_loader=None):
        for eps in range(epochs):
            print(f'Epoch {eps + 1}|{epochs}')
            errors = []

            for itr in trange(self.iterations, desc='Training'):
                error = self.step()
                errors.append(error)

            self.train_error.append(sum(errors) / len(errors))

            torch.save(self.model.state_dict(), os.path.join(
                self.checkpoint_path, f'{self.model_type}_autosave.pt'))

            if (eps + 1) % 10 == 0:
                torch.save(self.model.save_dict(), os.path.join(
                    self.checkpoint_path, f'params_ep_{eps + 1}.pt'))

            print(f'Average Train Error: {self.train_error[-1]}')

            ln = range(len(self.train_error))
            plt.figure(figsize=(10, 5))
            plt.title("Training Error")
            if val_loader is not None:
                self.val_error.append(self.validate(dataloader=val_loader,
                                                    show=False))
                plt.plot(ln, self.train_error, label='Average Train Error')
                plt.plot(ln, self.val_error, label='Average Validation Error')
            else:
                plt.plot(ln, self.train_error, label='Average Train Set Error')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            if HYAK:
                plt.savefig(os.path.join(self.checkpoint_path, 'errors.png'))
            else:
                plt.show()

            self.scheduler.step()

        @torch.no_grad()
        def validate(self, dataloader=None, show=True, TAU=0.1):
            errors = []
            self.model.eval()
            data = dataloader if dataloader is not None else self.data
            if self.model_type == 'taunet':
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

                errors.extend(error.cpu().numpy())

                if show:
                    self.show_validation_images(
                        fake_output_signal, real_output_signal)

            avg_err = np.mean(errors)

            print(f'Average Error: {avg_err}')

            return avg_err

        def show_validation_images(self, fake_output_signal,
                                   real_output_signal):
            show_images(torch.stack((fake_output_signal[0].cpu(
            ), real_output_signal[0].cpu(),
                torch.abs(fake_output_signal[0] - real_output_signal[0]
                          ).cpu()), dim=0), 3, 3)
            show_images(torch.permute(torch.stack((
                fake_output_signal[0].cpu(
                ), real_output_signal[0].cpu(),
                torch.abs(fake_output_signal[0] - real_output_signal[0]
                          ).cpu()), dim=0), (0, 1, 3, 4, 2)), 3, 3)
            show_images(torch.permute(torch.stack((
                fake_output_signal[0].cpu(
                ), real_output_signal[0].cpu(),
                torch.abs(fake_output_signal[0] - real_output_signal[0]
                          ).cpu()), dim=0), (0, 1, 4, 2, 3)), 3, 3)
