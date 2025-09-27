import torch.nn as nn
import torch
from model import utils


class Feature_extract(nn.Module):
    def __init__(self, devices, n_bands=48,depth=4,n_feats=64,dropout=0.0):
        super(Feature_extract, self).__init__()
        self.kernel_size = 3               
        self.devices = devices              
        self.n_bands = n_bands           
        self.n_feats = n_feats             
        self.depth = depth                 
        self.act = nn.ReLU(True)            
        self.bn = False
        self.conv_first = utils.default_conv(self.n_bands, self.n_feats, self.kernel_size)
        self.rcab1 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
       
    def forward(self, x):

        x = self.conv_first(x)
        feat = self.rcab1(x)
        
        return feat
    

class Reconstruction(nn.Module):
    def __init__(self, devices, depth=4,n_feats=64,dropout=0.0):
        super(Reconstruction, self).__init__()
        self.kernel_size = 3         
        self.devices = devices         
        self.n_feats = n_feats       
        self.depth = depth              
        self.act = nn.ReLU(True)            
        self.bn = False

        self.rcab1 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab2 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab3 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab4 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab5 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab6 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab7 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab8 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
        self.rcab9 = utils.RCAB(self.n_feats,reduction=8,bias=False, bn=self.bn, act=self.act, res_scale=1,dropout=dropout)
       
    def forward(self, x):

        feat = self.rcab1(x)
        feat = self.rcab2(feat)
        feat = self.rcab3(feat)
        feat = self.rcab4(feat)
        feat = self.rcab5(feat)
        feat = self.rcab6(feat)
        feat = self.rcab7(feat)
        feat = self.rcab8(feat)
        feat = self.rcab9(feat)    
        return feat



class Uncertainty_Estimator(nn.Module):
    def __init__(self,n_feats=64 ):
        super().__init__()
        self.cross_attention = utils.Cross_Attention(dim=n_feats)
        self.attention1 = utils.ESSAttn(dim=n_feats)

    def forward(self, z_i,z_i_1):
        out = self.cross_attention(z_i,z_i_1)
        out = self.attention1(out)
        return out
    
class Spectrum_Dewarp(nn.Module):
    def __init__(self,bands_eve,n_feats=64 ):
        super().__init__()
        self.conv_lr_0 = utils.default_conv(bands_eve, n_feats, 3)
        self.conv_lr_pre = utils.default_conv(bands_eve, n_feats, 3)
        self.y_cal = utils.ResBlock(n_feats)
        self.dif_cal = utils.ResBlock(n_feats)
        self.dewarp = utils.RCAB(n_feats)

    def forward(self,y, z_i,z_i_1):
        z_i = self.conv_lr_0(z_i)
        z_i_1 = self.conv_lr_pre(z_i_1)
        out = self.y_cal(y) + self.dif_cal(z_i - z_i_1)
        out = self.dewarp(out)
        return out + y
    

class ASPU(nn.Module):
    def __init__(self, n_feats,norm=False,reduction = 8,bands_eve=4):
        super().__init__()
        self.uncertainty_estimator = Uncertainty_Estimator(n_feats=n_feats)
        self.gain_calculator = nn.Sequential(
            utils.ResBlock(num_features=n_feats,norm=norm),
            utils.ResBlock(num_features=n_feats,norm=norm),
        )
        self.cal_spa = nn.Sequential(
            nn.Conv2d(n_feats, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.cal_spec = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats*2, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.spectrum_dewarp = Spectrum_Dewarp(bands_eve,n_feats)
        
        self.conv = utils.default_conv(n_feats*2, n_feats, 3)
    def predict(self, y, x,x_1):
        z_predict = self.spectrum_dewarp(y,x,x_1)
        return z_predict
    def update(self, z_code, z_predict, K_gain):
        K_spa = K_gain[0]
        K_spec = K_gain[1]
        z_spa = (1 - K_spa) * z_code + K_spa * z_predict   
        z_spec = torch.cat([z_code, z_predict], dim=1)
        z_spec = K_spec * z_spec                          
        z_spec = self.conv(z_spec)

        return z_spec + z_spa
    def calc_gain(self, z_code,z_1):
        h_codes = self.uncertainty_estimator(z_code,z_1) 
        K_gain = self.gain_calculator(h_codes)
        K_spa = self.cal_spa(K_gain)
        K_spec = self.cal_spec(K_gain)
        return [K_spa, K_spec]


class Spec_alignment(nn.Module):
    def __init__(self, channels=64,scale=4,bands_eve=4,drop_path=0.0,dropout=0.0):
        super(Spec_alignment, self).__init__()
        self.conv_difference = utils.ConvBNReLU2D(channels,channels,kernel_size=3,padding=1)
        if scale == 2:
            self.Up_x0 = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size=3,padding=1),
                nn.ConvTranspose2d(channels,channels,kernel_size=2,stride=2),
                nn.PReLU(channels))
            self.Up_fuse = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size=3,padding=1),
                nn.ConvTranspose2d(channels,channels,kernel_size=2,stride=2),
                nn.PReLU(channels))
        elif scale == 4:
            self.Up_x0 = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size=3,padding=1),
                nn.ConvTranspose2d(channels,channels,kernel_size=4,stride=4),
                nn.PReLU(channels))
            self.Up_fuse = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size=3,padding=1),
                nn.ConvTranspose2d(channels,channels,kernel_size=4,stride=4),
                nn.PReLU(channels))
        elif scale == 8:
            self.Up_x0 = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size=3,padding=1),
                nn.ConvTranspose2d(channels,channels,kernel_size=8,stride=8),
                nn.PReLU(channels))
            self.Up_fuse = nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size=3,padding=1),
                nn.ConvTranspose2d(channels,channels,kernel_size=8,stride=8),
                nn.PReLU(channels))

        self.fuse_att = utils.IISA_window_BFR(dim=channels)  # 1.6w
        
    def forward(self, x_back,x_forth,x):
        x0 = self.Up_x0(x)
        x_fuse = torch.cat([x_back.unsqueeze(1), x_forth.unsqueeze(1), x.unsqueeze(1)], dim=1) 
        x_fuse = x_fuse.permute(0,1,3,4,2)                           
        x_fuse = self.fuse_att(x_fuse)
        x_fuse = self.Up_fuse(x_fuse)
        diff = x_fuse - x0

        diff = self.conv_difference(diff)
        x_fuse = x0 + diff

        return x_fuse