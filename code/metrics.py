"""
Created on December 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
"""
import os
import json
import SimpleITK as sitk


def get_modality(path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(
        path)).T[None, None, ...].astype(dtype=float)
    return image


def get_vals(max_id, path=None, FILE_PREFIX='STIFF_'):
    images = [get_modality(os.path.join(
        path, f'{FILE_PREFIX}{idx}.nii')) for idx in range(max_id + 1)]
    max_vals = [img.max() for img in images]
    min_vals = [img.min() for img in images]
    return {'MAX': max(max_vals), 'MIN': min(min_vals)}


def run(max_id, HYAK=False, FILE_PREFIX='STIFF_'):
    assert isinstance(
        max_id, int), f"max_id Type mismatch! Expecting int, got \
        {type(max_id)} instead!"

    path = '/gscratch/kurtlab/vvp/data/train' if HYAK\
        else '/home/agam/Downloads/ME599/train'
    result_dict = get_vals(max_id, path, FILE_PREFIX)

    with open(os.path.join(path, 'stiff.json'), 'w') as file:
        json.dump(result_dict, file)


if __name__ == '__main__':
    FILE_PREFIX = 'STIFF_'
    HYAK = False
    max_id = 43
    run(max_id, HYAK, FILE_PREFIX)
