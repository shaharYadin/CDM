import os
import copy 

import numpy as np
import torch

import torch.nn.functional as F
import wandb

from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from accelerate.utils import set_seed
from tqdm import tqdm
from datetime import datetime
from modules import EMA


class CDM:
    def __init__(self, accelerator, model, optimizer, ema_factor, batch_size, num_workers, val_freq=None, display_name=None, 
                 ckpt_folder=None, ckpt_file=None, noise_steps=1000, ce_factor=0.001, mse_factor=1, 
                 num_classes=0, num_sampling_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.model = model.to(self.device)
        self.cond_model = num_classes is not None
        self.num_classes = num_classes - 1 if num_classes is not None else None
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(self.device)
        self.ema = EMA(ema_factor)
        self.optimizer = optimizer
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.val_freq = val_freq
        self.beta_start = beta_start #0.0001
        self.beta_end = beta_end #0.02
        self.ce_factor = ce_factor
        self.mse_factor = mse_factor 
        self.noise_steps = noise_steps

        self.beta = torch.linspace(beta_start, beta_end, noise_steps, dtype=torch.float32).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value = 1.)
        self.alpha_hat = torch.cat((torch.tensor([1]).to(self.device), self.alpha_hat, torch.tensor([0]).to(self.device)))
        self.alpha = torch.cat((torch.tensor([1]).to(self.device), self.alpha))
        self.beta = torch.cat((torch.tensor([0]).to(self.device), self.beta))

        ## Flow matching
        self.sigma_min = 1e-4
        self.t = torch.linspace(0,1, self.noise_steps).to(self.device)

        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

        self.ckpt_folder = ckpt_folder
        self.ckpt_file = ckpt_file
        self.display_name = display_name 
        
        skip = self.noise_steps // num_sampling_steps
        self.seq = range(1, self.noise_steps+1, skip)
        self.seq_prev = [-1] + list(self.seq[:-1])
        self.seq_next = list(self.seq[1:]) + [self.noise_steps+1]
    
    def sample_timesteps(self, n, FM=False):
        if FM:
            t = torch.randint(low=0, high=self.noise_steps, size=(n,)).to(self.device)
        else:
            t = torch.randint(low=0, high=self.noise_steps+2, size=(n,)).to(self.device)
        return t
    
    def noise_images(self, x, t): 
        Ɛ = torch.randn_like(x).to(self.device)
        return torch.sqrt(self.alpha_hat)[t][:, None, None, None] * x + torch.sqrt(1-self.alpha_hat)[t][:, None, None, None] * Ɛ, Ɛ
    

    def train(self, train_dataset, val_dataset, save_ckpt_freq=50, sample_val_images=10, num_iterations=None, sampler='ddim'):
        set_seed(38, device_specific=True)
        if self.accelerator.is_main_process:
            ckpt_folder = str(datetime.now())
            os.mkdir(ckpt_folder)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True, persistent_workers=True)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    num_workers=self.num_workers, drop_last=True, shuffle=True, persistent_workers=True)
        
        lr_lambda = lambda step: (  min((step + 1) / (5000*self.accelerator.num_processes), 1)   # Warm up
                        * 0.1 ** (step // (200000*self.accelerator.num_processes))  # Step decay
                        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        best_val_loss = np.inf
        num_iters = 0 
        num_epochs = int((num_iterations/len(train_loader))*self.accelerator.num_processes) + 1 
        
        if val_dataset is None:
            self.model, self.ema_model, self.optimizer, train_loader, scheduler = self.accelerator.prepare(self.model, self.ema_model, self.optimizer, train_loader, scheduler)
        else:
            self.model, self.ema_model, self.optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(self.model, self.ema_model, self.optimizer, train_loader, val_loader, scheduler)
            

        for epoch in range(num_epochs):
            
            self.model.train()

            total_images = 0
            total_batches = 0
            total_mse_loss = 0 
            total_ce_loss = 0

            pbar = tqdm(train_loader, disable=not self.accelerator.is_main_process)
            for images, labels in pbar:
                image_shape = images.shape[2]

                t = self.sample_timesteps(images.shape[0]) 
                sigma = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]

                x_t, noise = self.noise_images(images, t)
                x_t.requires_grad = True
                if self.cond_model:
                    outputs = self.model(x_t, y=labels)
                else:
                    outputs = self.model(x_t)
                outputs = torch.cumsum(outputs, dim=1)

                y = torch.sum(outputs[:,-1] - torch.diag(outputs[:,t]))
                
                ce_loss = self.ce(outputs, t)

                if self.mse_factor != 0:
                    x_grad = torch.autograd.grad(y, x_t, create_graph=True)[0]
                else:
                    x_grad = 0 
                
                eps = sigma * (x_grad + x_t)
                mse_loss_noise = self.mse(eps, noise)  
                

                if self.mse_factor == 0:
                    loss = ce_loss 
                else:
                    loss = self.ce_factor * ce_loss + self.mse_factor * mse_loss_noise

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                scheduler.step()

                self.accelerator.log({'step': num_iters, 'lr': self.optimizer.param_groups[0]['lr']})
                pbar.set_postfix(loss=mse_loss_noise.item())

                self.ema.step_ema(self.ema_model, self.model, step_start_ema=5000)
                

                total_images += self.accelerator.gather(torch.tensor([t.size(0)]).to(self.device)).sum().item()
                total_batches += 1

                total_mse_loss += self.accelerator.gather(mse_loss_noise * t.size(0)).sum().item()
                total_ce_loss += self.accelerator.gather(ce_loss).sum().item()
    
                num_iters += 1
                
            total_mse_loss = total_mse_loss / total_images 
            total_ce_loss = total_ce_loss / (total_batches * self.accelerator.num_processes)

            self.accelerator.log({'Training MSE loss': total_mse_loss})
            self.accelerator.log({'Training cross-entropy loss': total_ce_loss})
            
            if self.accelerator.is_main_process:
                if val_dataset is None and total_mse_loss < best_val_loss:
                    best_val_loss = total_mse_loss
                    torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': best_val_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + '_best_loss'))
                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': total_mse_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + '_last_epoch'))
            
            if val_dataset is not None and self.val_freq is not None and epoch % self.val_freq == 0:

                total_images = 0
                total_batches = 0 
                total_mse_loss = 0
                total_ce_loss = 0 

                self.model.eval()

                total_log_likelihood = 0 
                for images, labels in tqdm(val_loader, disable=not self.accelerator.is_main_process):
                    t = self.sample_timesteps(images.shape[0])
                   
                    sigma = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
                    
                    x_t, noise = self.noise_images(images, t)
                    x_t.requires_grad = True

                    if self.cond_model:
                        uncond_indices = torch.rand(labels.shape) < 0.1 
                        labels[uncond_indices] = self.num_classes

                    outputs = self.ema_model(x_t, y=labels) if self.cond_model else self.ema_model(x_t)
                    outputs = torch.cumsum(outputs, dim=1)

                    y = torch.sum(outputs[:,-1] - torch.diag(outputs[:,t]))
                    ce_loss = self.ce(outputs, t)
                    x_grad = torch.autograd.grad(y, x_t, create_graph=False)[0]
                    
                    eps = sigma * (x_grad + x_t)
                    mse_loss_noise = self.mse(eps, noise)
                     
                    total_images += self.accelerator.gather(torch.tensor([t.size(0)]).to(self.device)).sum().item()
                    total_batches += 1
                    total_mse_loss += self.accelerator.gather(mse_loss_noise * t.size(0)).sum().item()
                    total_ce_loss += self.accelerator.gather(ce_loss).sum().item()
                    
                    #Likelihood Computation 
                    with torch.no_grad():
                        outputs = self.ema_model(images, y=labels) if self.cond_model else self.ema_model(images)
                        outputs = torch.cumsum(outputs, dim=1)
                
                    d = images.shape[1] * images.shape[2] * images.shape[3]
                    
                    log_likelihood = -(d/2)*np.log(2*np.pi) - 0.5*(torch.sum(images**2, dim=(1, 2, 3))) + (outputs[:,0] - outputs[:,-1])
                    log_likelihood /= d
                    total_log_likelihood +=self.accelerator.gather(log_likelihood).sum().item()


                val_mse_loss = total_mse_loss / total_images #total_batches
                val_ce_loss = total_ce_loss / (total_batches * self.accelerator.num_processes)
                nll = - (total_log_likelihood / total_images) / np.log(2) + 7 

                if self.accelerator.is_main_process:
                    if val_mse_loss < best_val_loss:
                        best_val_loss = val_mse_loss
                        torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': best_val_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + '_best_val_loss'))

                
                self.accelerator.log({'Validation MSE loss': val_mse_loss})
                self.accelerator.log({'Validation Cross-entropy loss': val_ce_loss})
                self.accelerator.log({'Validation NLL': nll})
               
            if epoch % sample_val_images == 0:
                
                sampled_imgs = self.sample(num_samples=10, image_shape=image_shape, sampler=sampler, validation=True)
                imgs_grid = make_grid(sampled_imgs, nrow=10)
                log_imgs = wandb.Image(imgs_grid)
                self.accelerator.log({"Sampled Images": log_imgs})
            if self.accelerator.is_main_process:
                if (epoch + 1) % save_ckpt_freq == 0:
                    torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': total_mse_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + f'_epoch{epoch}')) 
    
    def FM_train(self, train_dataset, val_dataset, save_ckpt_freq=50, sample_val_images=10, num_iterations=None):
        set_seed(38, device_specific=True)
        if self.accelerator.is_main_process:
            ckpt_folder = str(datetime.now())
            os.mkdir(ckpt_folder)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True, persistent_workers=True)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    num_workers=self.num_workers, drop_last=True, shuffle=True, persistent_workers=True)
        
        lr_lambda = lambda step: (  min((step + 1) / (5000*self.accelerator.num_processes), 1)   # Warm up
                        * 0.1 ** (step // (200000*self.accelerator.num_processes))  # Step decay
                        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        best_val_loss = np.inf
        num_iters = 0 
        num_epochs = int((num_iterations/len(train_loader))*self.accelerator.num_processes) + 1 
        if val_dataset is None:
            self.model, self.ema_model, self.optimizer, train_loader, scheduler = self.accelerator.prepare(self.model, self.ema_model, self.optimizer, train_loader, scheduler)
        else:
            self.model, self.ema_model, self.optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(self.model, self.ema_model, self.optimizer, train_loader, val_loader, scheduler)

        for epoch in range(num_epochs):
            
            self.model.train()

            total_images = 0
            total_batches = 0
            total_mse_loss = 0 
            total_ce_loss = 0

            pbar = tqdm(train_loader, disable=not self.accelerator.is_main_process)
            for images, labels in pbar:
                image_shape = images.shape[2]

                t = self.sample_timesteps(images.shape[0], FM=True) 
                t_0_1 = self.t[t][:,None,None,None]

                noise = torch.randn_like(images)

                x_t = t_0_1 * images + (1-(1-self.sigma_min)*t_0_1) * noise 

                x_t.requires_grad = True
                if self.cond_model:
                    outputs = self.model(x_t, y=labels)
                else:
                    outputs = self.model(x_t)
                outputs = torch.cumsum(outputs, dim=1)

                y = torch.sum(outputs[:,0] - torch.diag(outputs[:,t]))
                
                ce_loss = self.ce(outputs, t)

                if self.mse_factor != 0:
                    x_grad = torch.autograd.grad(y, x_t, create_graph=True)[0]
                else:
                    x_grad = 0 
                
                target = (1-(1-self.sigma_min)*t_0_1) * (x_grad + x_t)
                mse_loss_noise = self.mse(target, noise)

                if self.mse_factor == 0:
                    loss = ce_loss 
                else:
                    loss = self.ce_factor * ce_loss + self.mse_factor * mse_loss_noise

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                scheduler.step()

                self.accelerator.log({'step': num_iters, 'lr': self.optimizer.param_groups[0]['lr']})
                pbar.set_postfix(loss=mse_loss_noise.item())

                self.ema.step_ema(self.ema_model, self.model, step_start_ema=5000) 
                

                total_images += self.accelerator.gather(torch.tensor([t.size(0)]).to(self.device)).sum().item()
                total_batches += 1

                total_mse_loss += self.accelerator.gather(mse_loss_noise * t.size(0)).sum().item()
                total_ce_loss += self.accelerator.gather(ce_loss).sum().item()
                num_iters += 1
                
            total_mse_loss = total_mse_loss / total_images 
            total_ce_loss = total_ce_loss / (total_batches * self.accelerator.num_processes)

            self.accelerator.log({'Training MSE loss': total_mse_loss})
            self.accelerator.log({'Training cross-entropy loss': total_ce_loss})
            
            if self.accelerator.is_main_process:
                if val_dataset is None and total_mse_loss < best_val_loss:
                    best_val_loss = total_mse_loss
                    torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': best_val_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + '_best_loss'))
                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': total_mse_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + '_last_epoch'))
            
            if val_dataset is not None and self.val_freq is not None and epoch % self.val_freq == 0:

                total_images = 0
                total_batches = 0 
                total_mse_loss = 0
                total_ce_loss = 0 

                self.model.eval()

                idx = 0
                total_log_likelihood = 0 
                for images, labels in tqdm(val_loader, disable=not self.accelerator.is_main_process):

                    t = self.sample_timesteps(images.shape[0], FM=True)
                    t_0_1 = self.t[t][:,None,None,None]

                    noise = torch.randn_like(images)
                    x_t = t_0_1 * images + (1-(1-self.sigma_min)*t_0_1) * noise 
            
                    x_t.requires_grad = True

                    outputs = self.ema_model(x_t, y=labels) if self.cond_model else self.ema_model(x_t)
                    outputs = torch.cumsum(outputs, dim=1)

                    y = torch.sum(outputs[:,0] - torch.diag(outputs[:,t]))
                    ce_loss = self.ce(outputs, t)
                    x_grad = torch.autograd.grad(y, x_t, create_graph=False)[0]
                    
                    target = (1-(1-self.sigma_min)*t_0_1) * (x_grad + x_t)
                    mse_loss_noise = self.mse(target, noise)
                     
                    total_images += self.accelerator.gather(torch.tensor([t.size(0)]).to(self.device)).sum().item()
                    total_batches += 1
                    total_mse_loss += self.accelerator.gather(mse_loss_noise * t.size(0)).sum().item()
                    total_ce_loss += self.accelerator.gather(ce_loss).sum().item()
                    
                    #Likelihood Computation 
                    with torch.no_grad():
                        outputs = self.ema_model(images, y=labels) if self.cond_model else self.ema_model(images)
                        outputs = torch.cumsum(outputs, dim=1)
                
                    d = images.shape[1] * images.shape[2] * images.shape[3]
                    log_likelihood = -(d/2)*np.log(2*np.pi) - 0.5*(torch.sum(images**2, dim=(1, 2, 3))) + (outputs[:,-1] - outputs[:,0])
                    log_likelihood /= d
                    
                    total_log_likelihood +=self.accelerator.gather(log_likelihood).sum().item()                        

                    idx += 1 
                val_mse_loss = total_mse_loss / total_images 
                val_ce_loss = total_ce_loss / (total_batches * self.accelerator.num_processes)
                nll = - (total_log_likelihood / total_images) / np.log(2) + 7 

                if self.accelerator.is_main_process:
                    if val_mse_loss < best_val_loss:
                        best_val_loss = val_mse_loss
                        torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': best_val_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + '_best_val_loss'))

                
                self.accelerator.log({'Validation MSE loss': val_mse_loss})
                self.accelerator.log({'Validation Cross-entropy loss': val_ce_loss})
                self.accelerator.log({'Validation NLL': nll})
                
            if epoch % sample_val_images == 0:
                sampled_imgs = self.FM_sample(num_samples=10, image_shape=image_shape, validation=True)
                imgs_grid = make_grid(sampled_imgs, nrow=10)
                log_imgs = wandb.Image(imgs_grid)
                self.accelerator.log({"Sampled Images": log_imgs})
            if self.accelerator.is_main_process:
                if (epoch + 1) % save_ckpt_freq == 0:
                    torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.accelerator.unwrap_model(self.model.state_dict()),
                                    'ema_model_state_dict': self.accelerator.unwrap_model(self.ema_model.state_dict()),
                                    'optimizer_state_dict': self.accelerator.unwrap_model(self.optimizer.state_dict()),
                                    'loss': total_mse_loss,
                                    }, os.path.join(ckpt_folder, self.display_name + f'_epoch{epoch}')) 
    
                    
    def calc_nll(self, testset):

        self.model.to(self.device)
        ckpt = torch.load(os.path.join('.', self.ckpt_folder, self.ckpt_file), map_location=self.device)['ema_model_state_dict']
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
        self.model.eval()
        
        test_loader = DataLoader(testset, batch_size=self.batch_size,
                                    num_workers=self.num_workers, drop_last=True, shuffle=True, persistent_workers=True)
        total_log_likelihood = 0 
        num_images = 0 

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(images, labels) if self.cond_model else self.model(images)
                outputs = torch.cumsum(outputs, dim=1)

            d = images.shape[1] * images.shape[2] * images.shape[3]
            log_likelihood = -(d/2)*np.log(2*np.pi) - 0.5*(torch.sum(images**2, dim=(1, 2, 3))) + (outputs[:,0] - outputs[:,-1])
            log_likelihood /= d
            total_log_likelihood += torch.sum(log_likelihood).item()
            num_images += images.shape[0]
        
        nll = - (total_log_likelihood / num_images) / np.log(2) + 7 
        return nll 
   
    def denoiser(self, x_t, t, labels=None, w_cfg=1.25, validation=False):
        
        if self.cond_model and validation:
            labels = torch.arange(10, dtype=torch.int32, device=self.accelerator.device)
        num_samples = x_t.shape[0]
        sigma = torch.sqrt(1-self.alpha_hat[t]).item()

        if labels is None:
            if validation:
                outputs = self.ema_model(x_t)
            else:
                outputs = self.model(x_t)
        else:
            if w_cfg != 0:
                x_t = torch.cat((x_t,x_t), dim=0)
                x_t = x_t.detach()
                x_t.requires_grad = True
                labels = torch.cat((labels, labels), dim=0)
                labels[int(x_t.shape[0]/2):] = self.num_classes
            if validation:
                outputs = self.ema_model(x_t, y=labels)
            else:
                outputs = self.model(x_t, y=labels)
        outputs = torch.cumsum(outputs, dim=1)

        y = torch.sum(outputs[:,-1] - outputs[:,t])
        x_grad = torch.autograd.grad(y, x_t, create_graph=False)[0]

        eps = sigma * (x_grad + x_t)

        if labels is not None:
            if w_cfg!=0:
                cond_eps = eps[:num_samples]
                uncond_eps = eps[num_samples:]
                eps = (1+w_cfg) * cond_eps - w_cfg * uncond_eps
        
        return eps.detach()
    
   
    def sample(self, num_samples, image_shape, labels=None, w_cfg=1.25, sampler='ddpm', validation=False):
        
        if not validation:
            ckpt = torch.load(os.path.join('.', self.ckpt_folder, self.ckpt_file), map_location=self.device)['ema_model_state_dict']
            self.model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
            self.model.eval()

        if validation:
            self.ema_model.eval()
        input_channels = 3 
        x = torch.randn(num_samples, input_channels, image_shape, image_shape).to(self.device)
        

        ## For DPM-solver
        eps_p1 = None
        alpha_solver = torch.sqrt(self.alpha_hat)
        sigma_solver = torch.sqrt(1-self.alpha_hat)
        lambda_solver = torch.log(alpha_solver/sigma_solver)

        for idx, (t, s, t_next) in enumerate(tqdm(zip(reversed(self.seq), reversed(self.seq_prev), reversed(self.seq_next)), total=len(self.seq), leave=False)):
            x.requires_grad = True
           
            eps = self.denoiser(x, t, labels=labels, w_cfg=w_cfg, validation=validation) 
            
            alpha_hat = self.alpha_hat[t]
            alpha_hat_prev = self.alpha_hat[s]
            alpha = self.alpha[t]
            beta = self.beta[t]
            beta_tilde = ((1-alpha_hat_prev) / (1 - alpha_hat)) * beta
            noise = torch.randn_like(x)
            img_pred = (x - torch.sqrt(1-alpha_hat) * eps) / torch.sqrt(alpha_hat)

            if s != - 1:
                if sampler == 'ddpm':
                    x = ((torch.sqrt(alpha_hat_prev)*beta) / (1 - alpha_hat)) * img_pred + ((torch.sqrt(alpha)*(1-alpha_hat_prev))/(1-alpha_hat)) * x + torch.sqrt(beta_tilde)*noise
                elif sampler == 'ddim'  or (sampler == 'second_order_dpm_solver' and idx==0):
                    x = torch.sqrt(alpha_hat_prev) * img_pred + torch.sqrt(1-alpha_hat_prev) * eps 
                elif sampler == 'second_order_dpm_solver':
                    tm1, tp1 = s, t_next
                    m0, m1 = eps, eps_p1
                    lambda_tm1, lambda_t, lambda_tp1 = lambda_solver[tm1], lambda_solver[t], lambda_solver[tp1]
                    alpha_tm1, alpha_t = alpha_solver[tm1], alpha_solver[t]
                    sigma_tm1, sigma_t = sigma_solver[tm1], sigma_solver[t]
                    h, h_0 = lambda_tm1 - lambda_t, lambda_t - lambda_tp1
                    D0, D1 = m0, (1.0 / h_0) * (m0 - m1)
                    x = (
                    (alpha_tm1 / alpha_t) * x
                    - (sigma_tm1 * (torch.exp(h) - 1.0)) * D0
                    - (sigma_tm1 * (torch.exp(h) - h - 1.0)) * D1
                    )   
    
                else:
                    raise NotImplementedError()
                eps_p1 = eps.detach() 
                x = x.detach()
            else:
                x = img_pred 
            
        x = (x.clamp(-1, 1) + 1) / 2
        if not validation:
            x = (x * 255).type(torch.uint8)
        return x
    
    def FM_sample(self, num_samples, image_shape, validation=False, num_sampling_steps=50):
        
        if not validation:
            ckpt = torch.load(os.path.join('.', self.ckpt_folder, self.ckpt_file), map_location=self.device)['ema_model_state_dict']
            self.model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
            self.model.eval()

        if validation:
            self.ema_model.eval()
            
        input_channels = 3 
        x = torch.randn(num_samples, input_channels, image_shape, image_shape).to(self.device)

        dt = self.t[self.noise_steps//num_sampling_steps] - self.t[0]
        for t in tqdm(range(1, self.noise_steps, self.noise_steps//num_sampling_steps)):

            x.requires_grad = True 
            outputs = self.model(x)
            outputs = torch.cumsum(outputs, dim=1)
            y = torch.sum(outputs[:,0] - outputs[:,t])
            x_grad = torch.autograd.grad(y, x, create_graph=True)[0]
            t_0_1 = self.t[t]
            eps = (1-(1-self.sigma_min)*t_0_1) * (x_grad + x)
            x = x + (dt / t_0_1) * (x - eps)
            x = x.detach()

            
        x = (x.clamp(-1, 1) + 1) / 2
        if not validation:
            x = (x * 255).type(torch.uint8)
        return x
