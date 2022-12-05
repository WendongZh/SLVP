import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from core.model import CPL_base,CPL
import cv2
from core.utils import preprocess
from core.data_provider import datasets_factory
import itertools
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = configs.num_hidden
        self.num_layers = configs.num_layers
        networks_map = {
            'CPL_base': CPL_base.RNN,
            'CPL':CPL.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        if configs.model_name=='CPL':
            self.prediction_optimizer = Adam(itertools.chain(self.network.encoder.parameters(),self.network.decoder.parameters(),
            self.network.prior.parameters(),self.network.embed_class.parameters(),self.network.embed_data.parameters()), lr=configs.lr)
            self.recon_optimizer=Adam(itertools.chain(self.network.head_reconstructor.parameters(),self.network.prior_fp.parameters(),
            self.network.shared_encoder.parameters(),self.network.latent_encoder.parameters()),lr=configs.lr)
        else:
            self.optimizer=Adam(self.network.parameters(),lr=configs.lr)
        self.MSE_criterion = nn.MSELoss( )

    def save(self, itr, prefix = ''):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, prefix + 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=self.configs.device)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask,category):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.prediction_optimizer.zero_grad()
        self.recon_optimizer.zero_grad()
        category = torch.FloatTensor(category).to(self.configs.device)
        next_frames,loss,loss_pd,loss_kl,loss_cat,loss_recon = self.network(frames_tensor, mask_tensor,category)
        loss.backward()
        self.prediction_optimizer.step()
        self.recon_optimizer.step()
        return loss.detach().cpu().numpy(),loss_pd.detach().cpu().numpy(),loss_kl.detach().cpu().numpy()

    def CPL_train(self, pre_model,frames, mask,category,itr,is_replay=False):
        #torch.autograd.set_detect_anomaly(True)
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        category = torch.FloatTensor(category).to(self.configs.device)
        
        self.prediction_optimizer.zero_grad()
        self.recon_optimizer.zero_grad()
        next_frames,loss,loss_pd,loss_kl,loss_cat,loss_recon = self.network(frames_tensor, mask_tensor,category,is_train=True)
        loss.backward()
        self.prediction_optimizer.step()

        self.recon_optimizer.step()
        return loss.detach().cpu().numpy(),loss_pd.detach().cpu().numpy(),loss_kl.detach().cpu().numpy(),loss_cat.detach().cpu().numpy(),loss_recon.detach().cpu().numpy()
    

    def test(self, test_model,frames, mask,category):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        frames_tensor=frames_tensor.repeat(self.configs.num_samples,1,1,1,1)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        
        if self.configs.is_training:
            with torch.no_grad():
                next_frames = self.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,0,is_train=False,is_replay=False)[0]
                next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                loss=nn.MSELoss()(next_frames[:,:self.configs.input_length],frames_tensor[:,:self.configs.input_length])
                pred_cat=0
                
                for i in range(1,3):
                    next_frames = self.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,i,is_train=False,is_replay=False)[0]
                    next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                    loss_i=nn.MSELoss()(next_frames[:,:self.configs.input_length],frames_tensor[:,:self.configs.input_length])
                    if loss_i<loss:
                        loss=loss_i
                        pred_cat=i
                next_frames,right_cat_num = self.network(frames_tensor, mask_tensor,pred_cat, is_train=False)
        else:
            # test_model.load(self.configs.pretrained_model)
            with torch.no_grad():
                next_frames = test_model.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,0,is_train=False,is_replay=False)[0]
                next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                loss=nn.MSELoss()(next_frames,frames_tensor[:,:self.configs.input_length])
                pred_cat=0       
                for i in range(1,3):
                    next_frames = test_model.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,i,is_train=False,is_replay=False)[0]
                    next_frames=torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                    loss_i=nn.MSELoss()(next_frames,frames_tensor[:,:self.configs.input_length])
                    if loss_i<loss:
                        loss=loss_i
                        pred_cat=i
                    
            with torch.no_grad():
                next_frames,right_cat_num= test_model.network(frames_tensor, mask_tensor, pred_cat,is_train=False)
            
        return next_frames.detach().cpu().numpy(),right_cat_num.detach().cpu().numpy()


    def test_final_lpips(self, test_model,frames, mask,category):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        frames_tensor=frames_tensor.repeat(self.configs.num_samples,1,1,1,1)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        
        with torch.no_grad():

            loss_all = []
            for i in range(self.configs.num_classes): #
                next_frames = test_model.network(frames_tensor[:,:self.configs.input_length//2], mask_tensor,i,is_train=False,is_replay=False)[0]
                next_frames = torch.cat((frames_tensor[:,:1],next_frames[:,:self.configs.input_length-1]),1)
                loss_tmp = nn.MSELoss(reduction='none')(next_frames,frames_tensor[:,:self.configs.input_length])

                loss_tmp = torch.split(loss_tmp, self.configs.num_samples, 0)
                loss_tmp_label = []
                for item in loss_tmp:
                    loss_tmp_label.append(torch.mean(item))
                loss_tmp_label = torch.stack(loss_tmp_label, 0)
                loss_all.append(loss_tmp_label)

            loss_all = torch.stack(loss_all, 1)
            label_use = torch.argmin(loss_all, 1)
            label_use = label_use.float()
            print(label_use)
                
            next_frames,right_cat_num= test_model.network(frames_tensor, mask_tensor, label_use, is_train=False, final_generate=True)
            
            frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
            next_frames_forlpips = next_frames.permute(0, 1, 4, 2, 3).contiguous()

            batch = frames_tensor.size(0)
            timestep = frames_tensor.size(1)
            height = frames_tensor.size(3)
            width = frames_tensor.size(4)

            frames_forlpips = test_model.network.reshape_patchback(frames_tensor, batch, timestep, self.configs.patch_size, height, width)
            next_frames_forlpips = test_model.network.reshape_patchback(next_frames_forlpips, batch, timestep-1, self.configs.patch_size, height, width)

            frames_forlpips = torch.clip(frames_forlpips, 0.0, 1.0) * 2 -1
            next_frames_forlpips = torch.clip(next_frames_forlpips, 0.0, 1.0) * 2 -1

            frames_forlpips = frames_forlpips.repeat(1, 1, 3, 1, 1)
            next_frames_forlpips = next_frames_forlpips.repeat(1, 1, 3, 1, 1)

        return next_frames.detach().cpu().numpy(), right_cat_num.detach().cpu().numpy(), frames_forlpips[:, self.configs.input_length:], next_frames_forlpips[:, self.configs.input_length-1:]

    def test_relabel(self, test_model,frames, mask,category):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        frames_tensor=frames_tensor.repeat(self.configs.num_samples,1,1,1,1)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        if self.configs.model_name=='CPL_base':
            right_cat_num=torch.tensor(0)
            next_frames= self.network(frames_tensor, mask_tensor,category,is_train=False)[0]

            return next_frames.detach().cpu().numpy(),right_cat_num.detach().cpu().numpy()
        elif self.configs.model_name=='CPL':

            with torch.no_grad():

                loss_all = []
                for i in range(self.configs.num_classes): #
                    next_frames = self.network(frames_tensor, mask_tensor, i, is_train=False, is_replay=False)[0]
                    next_frames = torch.cat((frames_tensor[:,:1], next_frames), 1)
                    loss_tmp = nn.MSELoss(reduction='none')(next_frames[:, -1], frames_tensor[:, -1])
                    loss_tmp = torch.split(loss_tmp, self.configs.num_samples, 0)
                    loss_tmp_label = []
                    for item in loss_tmp:
                        loss_tmp_label.append(torch.mean(item))
                    loss_tmp_label = torch.stack(loss_tmp_label, 0)
                    loss_all.append(loss_tmp_label)

                loss_all = torch.stack(loss_all, 1)
                label_use = torch.argmin(loss_all, 1)
                label_use = label_use.float().detach().cpu().numpy()
            
            return label_use

    def parameters(self):
        return self.network.parameters()

    def named_parameters(self):
        return self.network.named_parameters()

    def load_state_dict(self, para):
        return self.network.load_state_dict(para)

    def state_dict(self):
        return self.network.state_dict()

