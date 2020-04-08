import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob
from collections import OrderedDict
class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                print('a')
                N_masks[f] = 255
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            import matplotlib.pyplot as plt
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            torchvision.utils.save_image(Fs.permute((1,0,2,3)),'fs.jpg')
            torchvision.utils.save_image(Ms.unsqueeze(2).view(-1,1,*Ms.shape[-2:]), 'ms.jpg')
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info


class CustomDataset(data.Dataset):
    def __init__(self,root,videos,single_object, transform, st=0,ed=None):
        self.root = root
        self.mask_root = os.path.join(root,'masks')
        self.image_root = os.path.join(root,'frames')

        self.videos = OrderedDict()
        self.transform  =transform
        self.st= st
        self.ed = ed
        with open(os.path.join(root,videos),'r') as lines:
            for line in lines:
                name = line.rstrip('\n')
                num_frame = len(glob.glob(os.path.join(self.image_root,name,'*.jpg')))
                mask = np.array(Image.open(os.path.join(self.mask_root,name,'00000.png')).convert('P'))
                num_objects = np.max(mask)
                shape = np.shape(mask)
                self.videos[name] = OrderedDict({'name':name,'num_frames': num_frame, 'num_objects':num_objects, 'shape':shape})
            self.K = 11
            self.single_object = single_object

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item ):
        video = list(self.videos.items())[item][1]
        name, num_frames,num_objects, shape = [v for k,v in video.items()]
        frames = []
        # get frames
        for i in range(num_frames):
            f = self.transform(Image.open(os.path.join(self.image_root,name,f'{i:05d}.jpg')))
            frames.append(f)
        frames = torch.stack(frames,1)
        if self.st:
            if self.ed:
                frames = frames[:,self.st:self.ed]
                num_frames = self.ed - self.st
            else:
                frames = frames[:,self.st:]
                num_frames = num_frames - self.st
        else:
            if self.ed:
                frames = frames[:,:self.ed]
                num_frames = self.ed
        video['num_frames'] = num_frames
        # get first mask
        m = self.transform(Image.open(os.path.join(self.mask_root,name,f'{self.st:05d}.png')).convert('P'))
        # B obj f h w
        mm = torch.zeros(num_frames,*m.shape[1:],dtype=torch.uint8)
        if self.single_object:
            mm[0] = m > 0
            Ms = torch.stack([(mm == k).to(torch.uint8) for k in range(self.K)]).float()
            num_objects = torch.LongTensor([int(1)])
            # torchvision.utils.save_image(frames.permute((1, 0, 2, 3)), 'ffs.jpg')
            # torchvision.utils.save_image(Ms.unsqueeze(2).view(-1, 1, *Ms.shape[-2:]), 'mms.jpg')
            return frames, Ms, num_objects, video
        # else:
        #     Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        #     num_objects = torch.LongTensor([int(self.num_objects[video])])
        #     return Fs, Ms, num_objects, info





if __name__ == '__main__':
    pass
