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
import os
import json
import random
import torch

from run import train, infer, get_optimal_params
from dataloader import val_dataloader, test_dataloader


torch.set_printoptions(precision=9)

# 'highest', 'high', 'medium'. 'highest' is slower but accurate while 'medium'
#  is faster but less accurate. 'high' is preferred setting. Refer:
#  https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision('medium')

# 'True' = faster but less accurate, 'False' = Slower but more accurate
#  has to be set to True if precision is high or medium
torch.backends.cuda.matmul.allow_tf32 = True

# 'True' = faster but less accurate, 'False' = Slower but more accurate
#  has to be set to True if presision is high or medium
torch.backends.cudnn.allow_tf32 = True

SEED = 64

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_ID = 43


def get_params():
    '''
    Load parameters from a JSON file.
    '''
    with open('.logs/parameters.json', 'r') as json_file:
        return json.load(json_file).values()


def run_model(learning_rate, epochs, reduction_factor, model_type,
              is_hyak=False):
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
    model_type : string
        Model that needs to be loaded.

    Returns
    -------
    None.

    '''
    assert type(model_type) is str, 'model_type must be of type string'
    assert type(epochs) is int, 'epochs must be of type int'
    assert type(reduction_factor) is int, 'reduction_factor must of type int'
    assert type(learning_rate) is float, 'learning_rate must be of type float'

    assert reduction_factor % 2 == 0, 'reduction_factor must me multiple of 2'
    assert reduction_factor > 0, 'reduction_factor must greater than 0'
    assert epochs > 0, 'epochs must greater than 0'

    path = '.plugins/parameters'
    operations = {
        '1': lambda: train(path, learning_rate=learning_rate, epochs=epochs,
                           n=reduction_factor, model_type=model_type,
                           max_id=MAX_ID),
        '2': lambda: infer(path, dataloader=val_dataloader(),
                           n=reduction_factor, model_type=model_type),
        '3': lambda: infer(path, dataloader=test_dataloader(),
                           n=reduction_factor, model_type=model_type)}

    while True:
        mode = '1' if is_hyak else input('Train(1), Validate(2), or Test(3): ')
        action = operations.get(mode)
        if action:
            action()
            break
        else:
            print(f'Error: {mode} is not a valid option. Please try again.')


def main(mode=0):
    if mode == 0:
        print('Running model...')
        run_model(*get_params())
    elif mode == 1:
        print('Computing Hyperparamaters...  (this may take a while)')
        params = list(get_params())
        param_search = get_optimal_params(params[-2], params[-3])
        optimal_params = param_search[0][-1]
        print(f'lr = {optimal_params[0]}\n \
              lambda1 = {optimal_params[1]}\n \
                  lambda2 = {optimal_params[2]}\n \
                      lambda3 = {optimal_params[3]}\n \
                          lambda4 = {optimal_params[4]}\n \
                              stepsize = {optimal_params[5]}\n \
                                  gamma = {optimal_params[6]}')
    else:
        with open('LICENSE', 'r') as file:
            contents = file.read()
        print(contents, '\n')


if __name__ == '__main__':
    path = os.path.abspath(__file__)[:-13]
    os.chdir(path)
    main(2)

    try:
        main(0)
        print(' \u2714 task completed successfully')
    except Exception:
        print(' \u274c task failed')
        main(1)
