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
from dataloader import train_dataloader, val_dataloader, test_dataloader


def create_path(hyak):
    return '/gscratch/kurtlab/vvp/code' if hyak else \
           '/home/agam/Documents/git-files/virtual-virtual-palpation/code/'


def train(path, learning_rate=1E-4, epochs=500, hyak=True, n=4):
    module = Trainer(checkpoint_path=path,
                     dataloader=train_dataloader(HYAK=hyak),
                     learning_rate=learning_rate, n=n)
    module.optimize(epochs=epochs, HYAK=hyak,
                    val_loader=val_dataloader(HYAK=hyak))
    module.validate(val_dataloader(HYAK=hyak))


def infer(path, dataloader, n=4):
    module = Trainer(checkpoint_path=path,
                     dataloader=dataloader, n=n)
    module.validate(dataloader)


if __name__ == '__main__':
    lr = 1E-4
    eps = 2000
    n = 2
    hyak = False
    path = create_path(hyak)

    while True:
        mode = input('Train(1), Validate(2), or Test(3): ')

        if mode == '3':
            infer(path, dataloader=test_dataloader(HYAK=hyak), n=n)
            break
        elif mode == '2':
            infer(path, dataloader=val_dataloader(HYAK=hyak), n=n)
            break
        elif mode == '1':
            train(path, learning_rate=lr, epochs=eps, hyak=hyak, n=n)
            break
        else:
            print(f'Error: {mode} is not a valid option. Please try again.')
