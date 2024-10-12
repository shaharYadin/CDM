import os

import torch
import torchvision.transforms as transforms
import yaml
from torchvision.datasets import CIFAR10

from CustomImageDataset import *
from accelerate import Accelerator

import wandb

from cdm import CDM
from utils import plot_images
from models.diffusion import DDPM_Unet



def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    #general config
    mode = config['general']['mode']
    dataset = config['general']['dataset']
    dataset_path = config['general']['dataset_path']
    model_arch = config['general']['model_arch']
    num_classes_cond = config['general']['num_classes_cond']  
    if num_classes_cond is not None:
        num_classes_cond += 1 #+1 for classifier free guidance
    beta_start = config['general']['beta_start']
    beta_end = config['general']['beta_end']
    noise_steps = config['general']['noise_steps']

    
    #Training config
    num_iterations = config['training']['num_iterations']
    lr = config['training']['lr']
    batch_size = config['training']['batch_size']
    val_freq = config['training']['val_freq']
    save_ckpt_freq = config['training']['save_ckpt_freq']
    sample_val_images = config['training']['sample_val_images']
    num_workers = config['training']['num_workers']
    ce_factor = config['training']['ce_factor']
    mse_factor = config['training']['mse_factor']
    ema_factor = config['training']['ema_factor']

    #Sampling config  
    ckpt_folder = config['sampling']['ckpt_folder']
    ckpt_file = config['sampling']['ckpt_file']
    num_samples = config['sampling']['num_samples']
    num_sampling_steps = config['sampling']['num_sampling_steps']
    image_shape = config['sampling']['image_shape']
    sampler = config['sampling']['sampler']
    labels = config['sampling']['labels']
    w_cfg = config['sampling']['w_cfg']

    

    #####################Dataset##########################

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset == 'CIFAR10':
        trainset = CIFAR10(root=os.path.join('.', 'datasets'),
                            train=True,
                            transform=transform,
                            download=True)
        valset = CIFAR10(root=os.path.join('.', 'datasets'),
                            train=False,
                            transform=transform,
                            download=True)    
        train_res = 32
        train_val_set = None

    elif dataset == 'celebA_64':
        train_res = 64 
        data_root = dataset_path
        train_val_set = CelebAUnconditional(data_root, image_size=[train_res, train_res])
    else:
        raise NotImplementedError()
    

    #####################Model##########################
    in_channels =  3
    out_channels = noise_steps if (mode == 'FM_training' or mode == 'FM_sampling') else noise_steps + 2
    if model_arch == 'ddpm_unet_small':
        model = DDPM_Unet(in_channels=in_channels, out_channels=out_channels, channels=128, image_size=train_res, resamp_with_conv=True, ch_mult=[1, 2, 2, 2], num_res_blocks=2, attn_resolutions=[16,], dropout=0.1, num_classes=num_classes_cond)
    elif model_arch == 'ddpm_unet_large':
        model = DDPM_Unet(in_channels=in_channels, out_channels=out_channels, channels=128, image_size=train_res, resamp_with_conv=True, ch_mult=[1, 2, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16,], dropout=0.1, num_classes=num_classes_cond)
    else:
        raise NotImplementedError()
    
    ####################Optimizer#########################
    optimizer = torch.optim.Adam(model.parameters() ,lr=lr)
    

    accelerator = Accelerator(log_with="wandb")

    wandb.login()
    display_name = f'mode={mode}_dataset={dataset}_model_arch={model_arch}_noise_steps={noise_steps}_num_classes_cond={num_classes_cond}'
    accelerator.init_trackers(
        project_name="CDM",
        config=config
    )

    
    cdm = CDM(accelerator=accelerator, model=model, optimizer=optimizer, ema_factor=ema_factor, batch_size=batch_size, num_workers=num_workers, 
                val_freq=val_freq, display_name=display_name, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file, noise_steps=noise_steps,
                ce_factor=ce_factor, mse_factor=mse_factor, num_classes=num_classes_cond, num_sampling_steps=num_sampling_steps, beta_start=beta_start, beta_end=beta_end)
    

    if mode == 'training':
        if train_val_set is not None:#For this dataset, training and validation have already splited 
            if val_freq is not None:
                val_size = int(len(train_val_set) * 0.1)
                train_size = len(train_val_set) - val_size
                trainset, valset = torch.utils.data.random_split(train_val_set, [train_size, val_size])
            else:
                trainset = train_val_set
                valset = None 
        cdm.train(train_dataset=trainset, val_dataset=valset, save_ckpt_freq=save_ckpt_freq, sample_val_images=sample_val_images, num_iterations=num_iterations, sampler=sampler)
        accelerator.end_training()
    
    elif mode == 'FM_training':
        if train_val_set is not None:#For this dataset, training and validation have already splited 
            if val_freq is not None:
                val_size = int(len(train_val_set) * 0.1)
                train_size = len(train_val_set) - val_size
                trainset, valset = torch.utils.data.random_split(train_val_set, [train_size, val_size])
            else:
                trainset = train_val_set
                valset = None 
        cdm.FM_train(train_dataset=trainset, val_dataset=valset, save_ckpt_freq=save_ckpt_freq, sample_val_images=sample_val_images, num_iterations=num_iterations)
        accelerator.end_training()
    
    elif mode == 'sampling':
        if num_classes_cond is not None and labels is not None:
            labels = torch.tensor([labels] * num_samples).to(accelerator.device)
        else:
            labels = None
        sampled_imgs = cdm.sample(num_samples=num_samples, image_shape=image_shape, labels=labels, w_cfg=w_cfg, sampler=sampler, validation=False)
        plot_images(sampled_imgs, figsize=(40,4))
    
    elif mode == 'FM_sampling':
        sampled_imgs = cdm.FM_sample(num_samples=num_samples, image_shape=image_shape, validation=False, num_sampling_steps=num_sampling_steps)
        plot_images(sampled_imgs, figsize=(40,4))
    
    elif mode == 'likelihood_eval':
        nll = cdm.calc_nll(testset=valset)
        print(f'negative log likelihood: {nll}')

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
