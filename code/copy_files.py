"""
Created on December 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
"""
import shutil
import os

src = r'/home/agam/Downloads/ME599/RAW/'
dst = '/home/agam/Downloads/ME599/train/'
src_files = ['ss_MREspace-T1.nii',
             '1p1_MRE_magnitude_fixed.nii',
             'ss_MREspace-adc.mif.nii',
             'ss_MREspace-fa.mif.nii',
             'ss_MREspace-rd.mif.nii',
             'ss_1p1_MRE_complexstiff_gadg_denoised_50Hz_stiffness.nii']
dst_files = ['T1',
             'T2',
             'DTI1',
             'DTI2',
             'DTI3',
             'STIFF']

idx = 0


def copy_file(src, dst, src_name, dst_name, idx):
    shutil.copy(os.path.join(src, src_name),
                os.path.join(dst, f"{dst_name}_{idx}.nii"))


for root, _, _ in os.walk(src):
    if root[-3:] == 'MRE':
        print(root)
        for i, src_file in enumerate(src_files):
            copy_file(root, dst, src_file, dst_files[i], idx)
        idx += 1
