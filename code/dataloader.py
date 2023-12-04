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
import SimpleITK as sitk
import torch
from torchio.transforms import RandomFlip, RandomAffine

from utils import norm, pad3d, show_images, plt

SIZE = 32 * 6


def rand_augment(x):
    flip = RandomFlip()
    affine = RandomAffine(image_interpolation='nearest', isotropic=False,
                          degrees=360, translation=20, scales=(0.8, 1.2))
    x_ = flip(affine(x))
    return x_


def augment_batch(x):
    for i in range(x.shape[0]):
        x[i] = rand_augment(x[i])
    return x


def get_modality(path, nrm=True):
    image = sitk.GetArrayFromImage(sitk.ReadImage(path)).T[None, None, ...]
    image = torch.from_numpy(image.astype(dtype=float))
    if nrm:
        image = norm(image)
    return image


def get_modalities(path, idx, nrm=True):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        t1_task = executor.submit(get_modality, os.path.join(
            path, 'T1_' + str(idx) + '.nii'), nrm)
        t2_task = executor.submit(get_modality, os.path.join(
            path, 'T2_' + str(idx) + '.nii'), nrm)
        d1_task = executor.submit(get_modality, os.path.join(
            path, 'DTI1_' + str(idx) + '.nii'), nrm)
        d2_task = executor.submit(get_modality, os.path.join(
            path, 'DTI2_' + str(idx) + '.nii'), nrm)
        d3_task = executor.submit(get_modality, os.path.join(
            path, 'DTI3_' + str(idx) + '.nii'), nrm)
        stiff_task = executor.submit(get_modality, os.path.join(
            path, 'STIFF_' + str(idx) + '.nii'), nrm)

    t1 = t1_task.result()
    mask = torch.where(t1 > 0, 1., 0.)
    t2 = t2_task.result() * mask
    d1 = d1_task.result() * mask
    d2 = d2_task.result() * mask
    d3 = d3_task.result() * mask
    sf = stiff_task.result() * mask

    return t1, t2, d1, d2, d3, sf


def load_patient(path, idx, nrm=True):
    t1, t2, d1, d2, d3, sf = get_modalities(path, idx, nrm)
    out = pad3d(torch.cat((t1, t2, d1, d2, d3, sf), dim=1), (SIZE, SIZE, SIZE))
    return out


def load_batch_dataset(path, idx_list):
    data = [load_patient(path, idx) for idx in idx_list]
    return torch.cat(data, dim=0)


class train_dataloader():
    def __init__(self, batch=1, max_id=43, post=False,
                 augment=True, HYAK=True, aug_thresh=0.05):
        self.augment = augment
        self.aug_thresh = aug_thresh
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
        self.idx = random.sample(range(0, sample_len), sample_len)

    def load_batch(self, post=False):
        if self.Flag:  # only runs the first time
            self.randomize()
            self.Flag = False

        if self.id + self.batch > self.max_id:
            if self.id < self.max_id:
                batch_raw = load_batch_dataset(self.path, self.idx[self.id:])
            elif self.id == self.max_id:
                batch_raw = load_batch_dataset(
                    self.path, self.idx[self.id:self.id + 1])
            self.id = 0
            self.randomize()
            if self.post:
                print('Dataset re-randomized...', self.idx)
        else:
            batch_raw = load_batch_dataset(
                self.path, self.idx[self.id:self.id + self.batch])
            self.id += self.batch

        if self.augment and random.uniform(0, 1) > self.aug_thresh:
            batch_raw = augment_batch(batch_raw)

        return (batch_raw[:, 0:5], batch_raw[:, 5:6])


class val_dataloader():
    def __init__(self,
                 pid=[49, 50, 51, 52, 53], batch=1, HYAK=True):
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

        return (batch_raw[:, 0:5], batch_raw[:, 5:6])


class test_dataloader():
    def __init__(self,
                 pid=[44, 45, 46, 47, 48], batch=1, HYAK=True):
        self.path = '/gscratch/kurtlab/vvp/data/test' if HYAK \
            else '/home/agam/Downloads/ME599/test'
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

        return (batch_raw[:, 0:5], batch_raw[:, 5:6])


if __name__ == '__main__':
    a = train_dataloader(HYAK=False, post=True, augment=False)
    for i in range(43):
        x = a.load_batch()
        show_images(torch.cat(x, dim=1).view(6, 1, SIZE, SIZE, SIZE), 6, 3)
        if (i+1) % 43 == 0:
            print('.......................')
