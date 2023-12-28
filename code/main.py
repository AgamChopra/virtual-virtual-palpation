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
import json

from run import train, infer, create_path
from dataloader import val_dataloader, test_dataloader


def get_params():
    '''
    Load parameters from a JSON file.
    '''
    with open('./parameters.json', 'r') as json_file:
        return json.load(json_file).values()


def run_model(learning_rate, epochs, reduction_factor, is_hyak, model_type):
    '''
    Run the model based on user input.

    Parameters
    ----------
    learning_rate : float
        Learning rate of the model.
    epochs : int
        Number of epochs to iterate.
    reduction_factor : int, multiple of 2 till 64
        Number of parameaters/channels in each layer/block.
    is_hyak : bool
        If code is running on HYAK or not.
    model_type : string
        Model that needs to be loaded.

    Returns
    -------
    None.

    '''
    assert type(model_type) == str, 'model_type must be of type string'
    assert type(is_hyak) == bool, 'is_hyak must be of type bool'
    assert type(epochs) == int, 'epochs must be of type int'
    assert type(reduction_factor) == int, 'reduction_factor must of type int'
    assert type(learning_rate) == float, 'learning_rate must be of type float'

    assert reduction_factor % 2 == 0, 'reduction_factor must me multiple of 2'
    assert reduction_factor > 0, 'reduction_factor must greater than 0'
    assert epochs > 0, 'epochs must greater than 0'

    path = create_path(is_hyak)
    operations = {
        '1': lambda: train(path, learning_rate=learning_rate, epochs=epochs,
                           hyak=is_hyak, n=reduction_factor,
                           model_type=model_type),
        '2': lambda: infer(path, dataloader=val_dataloader(HYAK=is_hyak),
                           n=reduction_factor, model_type=model_type),
        '3': lambda: infer(path, dataloader=test_dataloader(HYAK=is_hyak),
                           n=reduction_factor, model_type=model_type)}

    while True:
        mode = '1' if is_hyak else input('Train(1), Validate(2), or Test(3): ')
        action = operations.get(mode)
        if action:
            action()
            break
        else:
            print(f'Error: {mode} is not a valid option. Please try again.')


if __name__ == '__main__':
    run_model(*get_params())
