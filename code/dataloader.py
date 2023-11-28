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
import concurrent.futures
import random
import nibabel as nib
import torch
from torchio.transforms import RandomFlip, RandomAffine

from utils import norm, pad3d, show_images


def rand_augment(x):
    flip = RandomFlip()
    affine = RandomAffine(image_interpolation='nearest',
                          degrees=180, translation=4)
    x_ = flip(affine(x))
    return torch.clip(x_, torch.min(x), torch.max(x))


def augment_batch(x):
    for i in range(x.shape[0]):
        x[i] = rand_augment(x[i])
    return x


def get_modality(path, nrm=True):
    image = torch.from_numpy(nib.load(path).get_fdata())[None, None, ...]
    if nrm:
        image = norm(image)
    return image


def get_modalities(path, idx, nrm=True):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        t1_task = executor.submit(get_modality, os.path.join(
            path, 'T1_' + str(idx) + '.nii'), nrm)
        diff_task = executor.submit(get_modality, os.path.join(
            path, 'diff_' + str(idx) + '.nii'), nrm)
        mre_task = executor.submit(get_modality, os.path.join(
            path, 'stiff_' + str(idx) + '.nii'), nrm)

    t1 = t1_task.result()
    diff = diff_task.result()
    mre = mre_task.result()

    return t1, diff, mre


def load_patient(path, idx, nrm=True):
    t1, diff, mre = get_modalities(path, idx, nrm)
    out = pad3d(torch.cat((t1, diff, mre), dim=1), (180, 180, 180))
    return out


def load_batch_dataset(path, idx_list):
    data = [load_patient(path, idx) for idx in idx_list]
    return torch.cat(data, dim=0)


class train_dataloader():
    def __init__(self, batch=1, max_id=1, post=False,
                 augment=True, HYAK=True):
        self.augment = augment
        self.max_id = max_id
        self.id = 0
        self.batch = batch
        self.idx = None
        self.Flag = True
        self.post = post
        self.path = '/gscratch/kurtlab/vvp/data/train' if HYAK \
            else '/home/agam/Downloads/ME599/train'

    def randomize(self):
        sample_len = self.max_id + 1
        self.idx = random.sample(range(0, self.max_id + 1), sample_len)

    def load_batch(self, post=False):
        if self.Flag:  # only runs the first time
            self.randomize()
            self.Flag = False

        max_id = self.max_id

        if self.id + self.batch > max_id:
            if self.id < max_id:
                batch_raw = load_batch_dataset(self.path, self.idx[self.id:])
            elif self.id == max_id:
                batch_raw = load_batch_dataset(
                    self.path, self.idx[self.id:self.id + 1])
            self.id = 0
            self.randomize()
            if self.post:
                print('Dataset re-randomized...')
        else:
            batch_raw = load_batch_dataset(
                self.path, self.idx[self.id:self.id + self.batch])
            self.id += self.batch

        if self.augment:
            batch_raw = augment_batch(batch_raw)

        return (batch_raw[:, 0:2], batch_raw[:, 2:3])


class val_dataloader():
    def __init__(self,
                 pid=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 13, 14, 15, 16], batch=1, HYAK=True):
        self.path = '/gscratch/kurtlab/vvp/data/val' if HYAK \
            else '/home/agam/Downloads/ME599/val'
        self.pid = pid
        self.id = 0
        self.max_id = len(pid)
        self.batch = 1

    def load_batch(self):
        if self.id >= self.max_id:
            self.id = 0

        ids = [idx for idx in self.pid[self.id:self.id + self.batch]]
        batch_raw = load_batch_dataset(self.path, ids)

        self.id += self.batch

        return (batch_raw[:, 0:2], batch_raw[:, 2:3])


if __name__ == '__main__':
    a = train_dataloader(HYAK=False, post=True)
    for i in range(100):
        x = a.load_batch()
        print(x[0].shape, x[1].shape)
        show_images(torch.cat(x, dim=1).view(3, 1, 180, 180, 180), 3, 3)
        if (i+1) % 2 == 0:
            print('.......................')
