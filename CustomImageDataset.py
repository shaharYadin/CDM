import os
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms, Lambda
from PIL import Image




class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self._class.name_ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebAUnconditional(Dataset):
    def __init__(self, data_root, image_size=[64, 64], DDM_training=False):
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        
        if DDM_training:
            self.tfs = transforms.Compose([
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.tfs = transforms.Compose([
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.imgs = [os.path.join(data_root, p) for p in np.sort(os.listdir(data_root))]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        
        img = self.tfs(Image.open(self.imgs[index]).convert('RGB'))
        label = 0 #We don't need the label 
        return img, label
