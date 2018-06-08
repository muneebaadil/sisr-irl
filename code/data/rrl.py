import pdb 
import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

def RRL(dataset, args, train=True): 
    
    class _RRL(dataset): 
        def __init__(self, args, train=True):
            super(_RRL, self).__init__(args, train)

    return _RRL(args, train)
    
    
# class Featmaps(data.Dataset):
#     def __init__(self, args, train=True):
#         self.args = args
#         self.train = train
#         self.split = 'train' if train else 'test'
#         self.scale = args.scale
#         self.idx_scale = 0
#         self.repeat = args.test_every // (args.n_train // args.batch_size)

#         self._set_filesystem(args.dir_data)
#         def _scan():
#             list_hr = []
#             list_lr = [[] for _ in self.dir_data]
#             idx_begin = 0 if train else args.n_train
#             idx_end = args.n_train if train else args.offset_val + args.n_val

#             for i in range(idx_begin + 1, idx_end + 1):
#                 filename = self._make_filename(i)
#                 list_hr.append(self._name_hrfile(filename))
                
#                 for idx, dir_data_ in enumerate(self.dir_data):
#                     list_lr[idx].append(self._name_lrfile(filename, dir_data_))

#             return list_hr, list_lr

#         self.images_hr, self.images_lr = _scan()

#     def _set_filesystem(self, dir_data):
#         self.input_ext = 'npy'
#         self.label_ext = 'png'
        
#         if type(dir_data)==str:
#             self.dir_data = [dir_data]
#         elif type(dir_data)==list: 
#             self.dir_data = dir_data

#         self.dir_residuals = self.args.dir_residuals

#     def _make_filename(self, idx):
#         return idx 

#     def _name_hrfile(self, filename):
#         return os.path.join(self.dir_residuals, '{}.{}'.format(filename, self.label_ext))

#     def _name_lrfile(self, filename, dir_data):
#         return os.path.join(dir_data, '{}.{}'.format(filename, self.input_ext))

#     def __getitem__(self, idx):
#         feats, label = self._load_file(idx)     
#         feats, label = self._get_patch(feats, label)
        
#         return common.np2Tensor(feats, label, self.args.rgb_range)

#     def _load_file(self, idx):
        
#         idx = self._get_index(idx)
#         label = self.images_hr[idx]
#         label = misc.imread(label)
        
#         feats = []

#         for dir_num in xrange(len(self.images_lr)): 
#             feat = np.load(self.images_lr[dir_num][idx])
#             feat = feat.transpose((1,2,0))
#             #print 'feat shape now = {}'.format(feat.shape)
#             feats.append(feat)
        
#         if len(self.images_lr) > 1:
#             return np.concatenate(feats,dim=-1), label
#         else:
#             return feats[0], label

#     def _get_patch(self, feats, label):
#         patch_size = self.args.patch_size
#         scale = self.scale[self.idx_scale]

#         if self.train: 
#             feats, label = common.get_patch(feats, label, patch_size, scale, False)
#             feats, label = common.augment(feats, label)
#         else: 
#             ih, iw, _ = feats.shape
#             label = label[0:ih * scale, 0:iw * scale, :]
        
#         return feats, label

#     def set_scale(self, idx_scale):
#         self.idx_scale = idx_scale

#     def __len__(self):
#         if self.train:
#             return len(self.images_hr) * self.repeat // self.args.superfetch
#         else:
#             return len(self.images_hr)

#     def _get_index(self, idx):
#         if self.train:
#             return idx % len(self.images_hr)
#         else:
#             return idx
