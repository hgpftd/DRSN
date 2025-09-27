import torch
import torch.nn as nn
import torch.nn.functional as F

from model import module,utils


class DRSN(nn.Module):
    def __init__(self, devices, scale,bands,depth=16,bands_eve=4,n_feats=64,norm=False):
        super(DRSN, self).__init__()

        self.kernel_size = 3        
        self.devices = devices          
        self.scale = scale                 
        self.bands = bands                 
        self.bands_eve = bands_eve         
        self.num_group = bands // bands_eve 
        self.pad = False                   
        self.depth = depth                  
        self.pad_bands = 0                 
        self.act = nn.ReLU(True)         

        if bands % self.bands_eve !=0 :
            self.pad = True                                   
            self.pad_bands = bands_eve - bands % bands_eve    
            self.num_group += 1

        self.feat_extract = module.Feature_extract(devices, n_bands=self.bands_eve,depth=self.depth,n_feats=n_feats,dropout=0.0)
        self.reconstruct = module.Reconstruction(devices, depth=self.depth,n_feats=n_feats,dropout=0.0)
        self.conv_last = utils.default_conv(n_feats, self.bands_eve,self.kernel_size)
        self.ASPU = module.ASPU(n_feats=n_feats,norm=norm)
        self.SA = module.Spec_alignment(n_feats,scale)
        
    def forward(self, x):

        out = []
        B,C,H,W =x.shape  
        # spectral padding
        if self.pad:
            padding = torch.zeros(B,self.pad_bands,H,W).to(self.devices)
            x=torch.cat([x,padding],1)

        index = 0
        for x_lr in torch.chunk(x, self.num_group, 1):     
            #--- feature extract ---#
            z = self.feat_extract(x_lr)

            #--- ASPU ---#
            if index :
                K_gain = self.ASPU.calc_gain(z,z_1)             
                z_predict = self.ASPU.predict(y,x_lr,x_1)       
                z_update = self.ASPU.update(z,z_predict,K_gain) 
            else :
                z_update = z
            x_1 = x_lr
            z_1 = z

            #--- reconstruction ---#
            y = self.reconstruct(z_update)

            #--- spectrum alignment ---#
            if index:
                if index==1:
                    sr = self.SA(y,now,y)
                else:
                    sr = self.SA(back,now,y)
                # feat -> sr_out
                sr = self.conv_last(sr)
                sr = sr + F.interpolate(lr,(H*self.scale,W*self.scale))
                out.append(sr)                   
                # upgrade back and forth
                back = now
                now = y
                if index == self.num_group-1 :    # last group
                    sr = self.SA(back,now,back)
                    sr = self.conv_last(sr)
                    sr = sr + F.interpolate(x_lr,(H*self.scale,W*self.scale))  
                    out.append(sr)
            else:
                now = y

            lr = x_lr
            index += 1
        
        # Merge:
        out = torch.cat(out[:],1)[:,0:C,:,:]  
        out = out + F.interpolate(x,(H*self.scale,W*self.scale))

        return out
    