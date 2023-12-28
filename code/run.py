"""
Created on December 2023
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
    '''
    TO DO...

    Parameters
    ----------
    hyak : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return '/gscratch/kurtlab/vvp/code' if hyak else \
           '/home/agam/Documents/git-files/virtual-virtual-palpation/code/'


def train(path, learning_rate=1E-4, epochs=500, hyak=True,
          n=4, model_type='unet'):
    '''
    TO DO...

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 1E-4.
    epochs : TYPE, optional
        DESCRIPTION. The default is 500.
    hyak : TYPE, optional
        DESCRIPTION. The default is True.
    n : TYPE, optional
        DESCRIPTION. The default is 4.
    model_type : TYPE, optional
        DESCRIPTION. The default is 'unet'.

    Returns
    -------
    None.

    '''
    module = Trainer(checkpoint_path=path, model=model_type,
                     dataloader=train_dataloader(HYAK=hyak),
                     learning_rate=learning_rate, n=n)
    module.optimize(epochs=epochs, HYAK=hyak,
                    val_loader=val_dataloader(HYAK=hyak))
    module.validate(val_dataloader(HYAK=hyak))


def infer(path, dataloader, n=4, model_type='unet'):
    '''
    TO DO...

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    dataloader : TYPE
        DESCRIPTION.
    n : TYPE, optional
        DESCRIPTION. The default is 4.
    model_type : TYPE, optional
        DESCRIPTION. The default is 'unet'.

    Returns
    -------
    None.

    '''
    module = Trainer(checkpoint_path=path, model=model_type,
                     dataloader=dataloader, n=n)
    module.validate(dataloader)
