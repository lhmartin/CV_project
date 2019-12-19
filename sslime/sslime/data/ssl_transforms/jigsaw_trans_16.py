#!/usr/bin/env python3

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF


class JIG_SAW_TRANS_16(object):

    def __init__(self, rot = 0):
            
        cls = 16
        self.rot = rot
   
        self.permutations = self.__retrive_permutations(cls)

        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])

    #def __getitem__(self, index):
    def __call__(self, sample):
        #framename = self.data_path + '/' + self.names[index]
       
        #print(sample)
        order = np.random.randint(len(self.permutations))
        
        img = sample["data"][0].convert('RGB')
        
        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9  
        orig_tiles = [None] * 9  
        
        data, labels = [], []
        
        orig_img = img
        
        if(self.rot == 1):
            img = TF.rotate(img, self.angles[int(self.permutations[order][9])])
        
            
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
           
            tile = img.crop(c.tolist())
            orig_tile = orig_img.crop(c.tolist())
            
            tile = self.__augment_tile(tile)
            orig_tile = self.__augment_tile(orig_tile)
                
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile
            orig_tiles[n] = norm(orig_tile)
        
        data = [tiles[int(self.permutations[order][t])] for t in range(9)]
        data = torch.stack(data, 0)
        
        sample["data"] = [data]
        sample["label"] = [int(order)]

        return sample


    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        
        for row in images_list:
            row = row.split('\n')
            file_names.append(row[0])
            labels.append(0)

        return file_names, labels

    def __retrive_permutations(self, classes):

        if(self.rot == 1):
            all_perm = np.load('permutations/permutations_rot_hamming_max_%d.npy' %(classes))
        else:
            all_perm = np.load('permutations/permutations_hamming_max_%d.npy' % (classes))
            
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')