# -*- coding: utf-8 -*-
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
import torch
import torch.nn as nn
from tqdm import trange
from matplotlib import pyplot as plt

from models import Unet
from utils import show_images, pix_error


# Dataloader must return a tuple (batch of input, batch of ground truth)
class Trainer(nn.Module):
    def __init__(self, checkpoint_path, dataloader, CH_IN=2, CH_OUT=1, n=1,
                 optimizer=torch.optim.AdamW, criterion=[nn.MSELoss()],
                 lambdas=[1.], learning_rate=1E-4, device='cuda'):
        super(Trainer, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = torch.compile(Unet(CH_IN, CH_OUT, n)).to(device)
        try:
            self.model.load_state_dict(torch.load(
                os.path.join(checkpoint_path, 'autosave.pt')))
        except Exception:
            print('paramerts failed to load from last run')
        self.data = dataloader
        self.iterations = (int(dataloader.max_id / dataloader.batch_size) +
                           (dataloader.max_id % dataloader.batch_size > 0))
        self.optimizer = optimizer(self.model.parameters(), learning_rate)
        self.criterion = criterion
        self.lambdas = lambdas
        self.train_error = []

    def step(self):
        self.model.train()
        self.optimizer.zero_grad()

        x = self.data.load_batch().type(torch.float)
        input_signal = x[0].to(self.device).detach()
        real_output_signal = x[1].to(self.device).detach()

        fake_output_signal = self.model(input_signal)

        error = sum([self.lambdas[i] * self.criterion[i](real_output_signal,
                    fake_output_signal) for i in range(len(self.criterion))])

        error.backward()
        self.optimizer.step()

        return error.item()

    def optimize(self, epochs=200, HYAK=False):
        for eps in range(epochs):
            print(f'Epoch {eps + 1}|{epochs}')
            errors = []

            for itr in trange(self.iterations):
                error = self.step()
                errors.append(error)

            self.train_error.append(sum(errors) / len(errors))

            torch.save(self.model.state_dict(), os.path.join(
                self.checkpoint_path, "autosave.pt"))

            print(f'Average Compound Error: {self.train_error[-1]}')

            plt.figure(figsize=(10, 5))
            plt.title("Training Error")
            plt.plot(self.train_error, label='Average Compound Error')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Average Error")
            plt.legend()
            if HYAK:
                plt.savefig(os.path.join(self.checkpoint_path, 'errors.png'))
            else:
                plt.show()

    @torch.no_grad()
    def validate(self, dataloader):
        errors = []
        self.model.eval()
        data = dataloader
        for _ in range(len(data.pid)):
            x = self.data.load_batch().type(torch.float)
            input_signal = x[0].to(self.device).detach()
            real_output_signal = x[1].to(self.device).detach()
            fake_output_signal = self.model(input_signal)
            show_images(torch.stack(
                (fake_output_signal[0].cpu(), real_output_signal[0].cpu()),
                dim=0), 2, 2)
            errors.append(
                pix_error(real_output_signal[0].cpu(),
                          fake_output_signal[0].cpu()))
            print('Error=', errors[-1])
        print('Avg. prediction error=', sum(errors)/len(errors))
