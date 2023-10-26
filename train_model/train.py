import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from loss_net import LossNet
from transformer_net import TransformerNet
from dataset import train_dataloader

import numpy as np

import time
import os

import utils.utils as utils

def train(train_image_dir, 
          style_image_dir,
          style_image_name,
          batch_size = 1, 
          num_epochs = 2, 
          learning_rate = 0.001,
          content_weight = 1,
          style_weight = 4e5,
          tv_weight = 0,
          log = True,
          img_log_freq = 100,
          console_log_freq = 1,
          checkpoint_freq = 2000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device to run
    tb = SummaryWriter() #to log
    u = utils.Utils(models_dir="models", images_dir="images") #utils functions

    #data preparation
    transform_list = A.Compose([A.Resize(height = 256, width = 256),
                    A.CenterCrop(height = 256, width = 256),
                    A.ToFloat(),
                    ToTensorV2()])
    
    train_loader = train_dataloader(train_image_dir = train_image_dir, train_transforms = transform_list, batch_size = batch_size, num_workers = 4)

    #models preparation
    transformer_net = TransformerNet().train().to(device)
    loss_net = LossNet(requires_grad=False).to(device)

    #optimizer
    opt = Adam(transformer_net.parameters(), lr = learning_rate)

    #get style image representation
    style_img = u.preprocess_img(os.path.join(style_image_dir,style_image_name), device = device, batch_size = batch_size)
    
    target_style_features = loss_net(style_img)

    target_style_representation = [utils.gram_matrix(feature) for feature in target_style_features]

    tb_content_loss, tb_style_loss, tb_tv_loss = [0., 0., 0.] #to log

    #training loop
    t = time.time()
    for epoch in range(num_epochs):
        print(f'\nEPOCH: {epoch+1}\n')
        for i, content_batch in enumerate(train_loader):
            content_batch.to(device)
            style_batch = transformer_net(content_batch)

            content_batch_features = loss_net(content_batch)
            style_batch_features = loss_net(style_batch)

            #content representation and content loss
            target_content_representation = content_batch_features.relu2_2
            content_representation = style_batch_features.relu2_2
            content_loss = content_weight * torch.nn.MSELoss(reduction='mean')(target_content_representation, content_representation)

            #style representation and style loss
            style_loss = 0.0
            style_representation = [utils.gram_matrix(x) for x in style_batch_features]
            for gram_gt, gram_hat in zip(target_style_representation, style_representation):
                style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt, gram_hat)
            
            style_loss /= len(target_style_representation)
            style_loss *= style_weight

            #total variation loss (force image smoothness)
            tv_loss = tv_weight*utils.total_variation(style_batch)

            #sum loss
            total_loss = content_loss + style_loss + tv_loss

            #backprop
            total_loss.backward()
            opt.step()
            opt.zero_grad()

            #logging
            tb_content_loss += content_loss.item()
            tb_style_loss += style_loss.item()
            tb_tv_loss += tv_loss.item()

            if log:
                tb.add_scalar('Loss/Content', content_loss.item(), len(train_loader)*epoch + i+1)
                tb.add_scalar('Loss/Style', style_loss.item(), len(train_loader)*epoch + i+1)
                tb.add_scalar('Loss/Total-Variation', tv_loss.item(), len(train_loader)*epoch + i+1)

                if i % img_log_freq == 0:
                    stylized =  utils.post_process_image(style_batch[0].detach().to('cpu').numpy())
                    stylized = np.moveaxis(stylized, 2, 0) #channel first
                    tb.add_image('stylized_img', stylized, len(train_loader) * epoch + i + 1)

            if console_log_freq is not None and i % console_log_freq == 0:
                print(f'time elapsed={(time.time()-t)/60:.2f}[min]|epoch={epoch + 1}|batch=[{i + 1}/{len(train_loader)}]|c-loss={tb_content_loss / console_log_freq}|s-loss={tb_style_loss / console_log_freq}|tv-loss={tb_tv_loss / console_log_freq}|total loss={(tb_content_loss + tb_style_loss + tb_tv_loss) / console_log_freq}')
                tb_content_loss, tb_style_loss, tb_tv_loss = [0., 0., 0.]

            if checkpoint_freq is not None and (i + 1) % checkpoint_freq == 0:
                training_state = {
                    "content_weight": content_weight,
                    "style_weight": style_weight,
                    "tv_weight": tv_weight,
                    "num_epochs": num_epochs,
                    "state_dict": transformer_net.state_dict(),
                    "optimizer_state": opt.state_dict()
                }

                ckpt_model_name = f"ckpt_style_{style_image_name.split('.')[0]}_cw_{str(content_weight)}_sw_{str(style_weight)}_tw_{str(tv_weight)}_epoch_{epoch}_batch_{i}.pth"
                torch.save(training_state, os.path.join('models', ckpt_model_name))

if __name__ == '__main__':
    train(train_image_dir="data/mscoco/train2014", 
          style_image_dir="style_images", 
          style_image_name="starry_night.jpeg",
          batch_size=4, 
          num_epochs=2,
          tv_weight=0.1)