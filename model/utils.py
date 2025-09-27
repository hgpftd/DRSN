import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from einops import rearrange


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act=None, norm=None,dropout=0.1):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()
        #self.drop = nn.Dropout2d(dropout)

    def forward(self, inputs):

        out = self.layers(inputs)  # Conv
        if self.norm is not None:
            out = self.norm(out)   # Norm
        if self.act is not None:
            out = self.act(out)    # Act
        #out = self.drop(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, num_features,norm=False):
        super(ResBlock, self).__init__()
       
        self.layers = nn.Sequential(
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, act='ReLU', padding=1,norm=norm),
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, padding=1)
        )
    def forward(self, inputs):
        return F.relu(self.layers(inputs) + inputs)  # F(x)+x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):    
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)    
        y = self.conv_du(y)    
        return x * y


class RCAB(nn.Module):
    def __init__(
        self, n_feat,reduction=8,
        bias=False, bn=False, act=nn.ReLU(True), res_scale=1,dropout=0.0):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3,1,1, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat)) 
            if i == 0: modules_body.append(act)             
            modules_body.append(nn.Dropout2d(dropout))
        modules_body.append(CALayer(n_feat, reduction))        
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):

        res = self.body(x).mul(self.res_scale)
        res += x            
        return res
    


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:   
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3,stride=1,padding=1,bias=True))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 4 * n_feats, 3,stride=1,padding=1,bias=True))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            print("Error: scale=",scale)
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Cross_Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):   
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) 

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)    
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y): 
        b, c, h, w = x.shape   

        kv = self.kv_dwconv(self.kv(y))   
        k, v = kv.chunk(2, dim=1)        
        q = self.q_dwconv(self.q(x))    
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature   
        attn = attn.softmax(dim=-1)                       
        
        out = (attn @ v)                                    

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2)
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = t1 + t2
        attn = self.ln(attn)
        attn = attn.transpose(1,2).view(b,c,h,w)
        return attn


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],C)
    return windows


def window_reverse(windows, window_size, B, H, W):
    x = windows.view(B, H // window_size[1], W // window_size[2], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class IISA_window_BFR(nn.Module):
    def __init__(self, dim=64, num_heads=8, window_size=(3,16,16), bias=False):   
        super(IISA_window_BFR, self).__init__()

        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, 3*dim, bias=bias)                     
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)                                
        self.softmax = nn.Softmax(dim=-1)     

    def forward(self, x):  

        b, d, h, w, c = x.shape   

        attn = window_partition(x,self.window_size)
        B_, N, C = attn.shape

        qkv = self.qkv(attn)            
        qkv = qkv.reshape(B_, N, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        quary_q = qkv[0]                     
        quary_q = quary_q[:,:,2*(N//3):N,:]   
        k = qkv[1]                           
        v = qkv[2]                          

        quary_q = quary_q * self.scale       
        attn = quary_q @ k.transpose(-2, -1)  

        attn = self.softmax(attn)    
        x = (attn @ v).transpose(1, 2).reshape(B_, N//3, c)

        x = x.view(-1, self.window_size[1], self.window_size[2], c)        
        x = window_reverse(x, self.window_size, b, h, w)
        x = x.permute(0,3,1,2)

        x = self.proj(x)   
        return x  
