import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

import math
import pathlib
import warnings
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageColor, ImageDraw, ImageFont

import os 

def plot_images(images, file_name='Sampled_imgs.png', folder=None, axis_off=True, figsize=(20,2)):
    plt.figure(figsize=figsize)
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    if axis_off:
        plt.axis('off')
    if folder is None:
        plt.savefig(file_name, format='png')
    else:
        plt.savefig(os.path.join(folder, file_name), format='png')


        
def normalize_imgs(imgs):
    #from [-1,1] to [0,1]
    normalized_imgs = (imgs + 1) / 2
    normalized_imgs = normalized_imgs.clamp(0,1)
    return normalized_imgs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
