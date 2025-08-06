import torch.nn as nn
import torch.nn.functional as F
import torch
from .vrwkv import Block

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, stride, dilation=1):
        super().__init__()
        self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.layer = nn.Conv3d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.norm = nn.BatchNorm3d(out_dim)  
        self.act = nn.ReLU(inplace=True)     

    def forward(self, x):
        add = self.reduction(x)
        out = self.layer(self.act(add))
        out_res = self.act(add + out)
        return out_res

def make_layers(in_dim, out_dim, kernel_size=3, padding=1, stride=1, dilation=1, downsample=False, blocks=2):
    layers = []
    if downsample:
        layers.append(nn.MaxPool3d(2))
    layers.append(ResBlock(in_dim, out_dim, kernel_size, padding, stride, dilation))
    for _ in range(1, blocks):
        layers.append(ResBlock(out_dim, out_dim, kernel_size, padding, stride, dilation))
    return nn.Sequential(*layers)

class Occbranch(nn.Module):
    def __init__(self, in_channels, bev_h, bev_w, bev_z):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        
        self.in_layer = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=1),  
            nn.Conv3d(64, 32, kernel_size=1),          
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm3d(32),                        
            nn.ReLU(inplace=True)                      
        )
        
        self.block = make_layers(32, 16, kernel_size=3, padding=1, stride=1, dilation=1, blocks=1)
        self.block_1 = make_layers(32, 32, kernel_size=3, padding=1, stride=1, dilation=1, downsample=True,blocks=1)
        self.block_2 = make_layers(32, 64, kernel_size=3, padding=1, stride=1, dilation=1, downsample=True, blocks=1)
        
        
        # RWKV blocks
        self.rwkv_block1 = Block(n_embd=32, n_layer=18, layer_id=0)
        self.rwkv_block2 = Block(n_embd=64, n_layer=18, layer_id=0)
        self.rwkv_block3 = Block(n_embd=128, n_layer=18, layer_id=0)

        self.reduction_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU()
        )
        self.reduction_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reduction_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU()
        )
        self.out2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1))
        self.out4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=1))
        self.out8 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=1))
        
    def forward(self, x):
        # x[B, 128, 128, 128, 16]
        out = self.in_layer(x)  #[B, 32, 128, 128, 16]
        res = self.block(out)#[B, 16, 128, 128, 16]
        
        res1 = self.block_1(out)  # [B, 32, 64, 64, 8]1/2
        res2 = self.block_2(res1)  # [B, 64, 32, 32, 4]1/4

        occ_logits_2 = self.out2(res)
        occ_logits_4 = self.out4(res1)
        occ_logits_8 = self.out8(res2)

        # BEV
        res_bev = res.permute(0, 1, 4, 2, 3)#[B, 16, 16, 128, 128]
        bev_res = self.reduction_1(res_bev.flatten(1, 2))# B, 32, 128, 128

        res1_bev = res1.permute(0, 1, 4, 2, 3)  # [B, 32, 8, 64, 64]
        bev_res1 = self.reduction_2(res1_bev.flatten(1, 2)) # B, 64, 64, 64
        
        res2_bev = res2.permute(0, 1, 4, 2, 3) #[B, 64, 4, 32, 32]
        bev_res2 = self.reduction_3(res2_bev.flatten(1, 2))#B, 128, 32, 32

        B, C, H, W = bev_res.shape
        patch_resolution = (H, W)
        bev_res = self.rwkv_block1(bev_res.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        B, C, H, W = bev_res1.shape
        patch_resolution = (H, W)
        bev_res1 = self.rwkv_block2(bev_res1.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)
        
        B, C, H, W = bev_res2.shape
        patch_resolution = (H, W)
        bev_res2 = self.rwkv_block3(bev_res2.permute(0, 2, 3, 1).reshape(B, H * W, C), patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)
        
        return dict(
            mss_bev_dense = [bev_res, bev_res1, bev_res2],mss_logits_list = [occ_logits_2, occ_logits_4, occ_logits_8]
        )
        