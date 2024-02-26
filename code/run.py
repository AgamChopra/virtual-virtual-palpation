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
from dataloader import train_dataloader, val_dataloader
from models import get_models
from hyperparameter_search import FetchBestHyperparameters


def train(path, learning_rate=1E-4, epochs=500,
          n=4, model_type='unet', max_id=43):
    '''
    TO DO...

    Parameters
    ----------
    # path : TYPE
        DESCRIPTION.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 1E-4.
    epochs : TYPE, optional
        DESCRIPTION. The default is 500.
    n : TYPE, optional
        DESCRIPTION. The default is 4.
    model_type : TYPE, optional
        DESCRIPTION. The default is 'unet'.

    Returns
    -------
    None.

    '''
    post = False
    batch = 1
    max_id = max_id
    workers = 2

    trn_dtl = train_dataloader(post=post, augment=False, max_id=max_id,
                               workers=workers, batch=batch)

    module = Trainer(checkpoint_path=path, model_type=model_type,
                     dataloader=trn_dtl, learning_rate=learning_rate, n=n)
    module.optimize(epochs=epochs, val_loader=val_dataloader())
    module.validate(val_dataloader())


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
    module = Trainer(checkpoint_path=path, model_type=model_type,
                     dataloader=dataloader, n=n)
    module.validate(dataloader)


# TO DO: run hyperparameter search
def get_optimal_params(model_type, reduction_factor):
    models = get_models(CH_IN=5, CH_OUT=1, n=reduction_factor)
    param_search = FetchBestHyperparameters(model=models[model_type],
                                            state_name=model_type)
    return param_search.search()
