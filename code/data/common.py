import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st
import torch
from scipy.signal import convolve2d
from skimage.util import view_as_windows

import torch
from torchvision import transforms

def random_patch_select(img,ih,iw,ip): 
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    
    return ix, iy

def gradient_patch_select(img,ih,iw,ip):
    
    hkernel = np.array([[-1,0,1],[-2,0,+2],[-1,0,1]])
    vkernel = hkernel.T
    
    def _convolve(img,k): 
        temp1 = convolve2d(img[:,:,0], k, mode='valid')
        temp2 = convolve2d(img[:,:,1], k, mode='valid')
        temp3 = convolve2d(img[:,:,2], k, mode='valid')
        return np.stack([temp1,temp2,temp3],axis=-1)
    
    h_edges_input = _convolve(img,hkernel)
    v_edges_input = _convolve(img,vkernel)
    edges_mag_input = np.sum((v_edges_input**2) + (h_edges_input**2),axis=-1)
    
    patches = view_as_windows(edges_mag_input, window_shape=ip, step=ip)
    n_patches_v, n_patches_h = patches.shape[:2]

    probs = patches.sum(axis=(-1,-2)) / float(patches.sum())
    
    idx = np.random.choice(np.arange(n_patches_v*n_patches_h),p=probs.flatten())
    
    iy, ix = idx // n_patches_h, idx % n_patches_h
    ix, iy = ix*ip, iy*ip
    
    return ix,iy

def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False,
              strategy='random'):
    
    ih, iw = img_in.shape[:2]

    p = scale if multi_scale else 1
    tp = p * patch_size
    ip = tp // scale

    if strategy=='random': 
        ix, iy = random_patch_select(img_in, ih, iw, ip)
    elif strategy=='gradient': 
        ix, iy = gradient_patch_select(img_in, ih, iw, ip)
        
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def set_channel(l, n_channel_in, n_channel_out):
    def _set_channel(img, n_channel):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(l[0], n_channel_in), _set_channel(l[1], n_channel_out)]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.0)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]
