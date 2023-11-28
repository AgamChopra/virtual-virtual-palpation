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
from train_utils import Trainer
from dataloader import train_dataloader, val_dataloader  # Add test dataset


def train_validate(path, learning_rate=1E-4, epochs=500, hyak=True, n=4):
    module = Trainer(checkpoint_path=path,
                     dataloader=train_dataloader(HYAK=hyak),
                     learning_rate=learning_rate, n=n)
    module.optimize(epochs=epochs, HYAK=hyak)
    module.validate(val_dataloader())


def infer(path):
    # Update for test dataset
    module = Trainer(checkpoint_path=path, dataloader=val_dataloader())
    module.validate(val_dataloader())


if __name__ == '__main__':
    path = ''
    lr = 1E-4
    eps = 200
    hyak = False
    mode = input('Train(train), Validate(val), or Test(test)')

    if mode == 'test':
        infer(path)
    elif mode == 'val':
        train_validate(path, learning_rate=1E-10, epochs=0, hyak=hyak)
    elif mode == 'train':
        train_validate(path, learning_rate=lr, epochs=eps, hyak=hyak)
    else:
        print('Error: {mode} is not implemented!')
