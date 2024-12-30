import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from .vmamba_xb import SS2D_xb,SS2D

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf


class BridgeLayer2_xb(nn.Module):
    def __init__(self, dims=48):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims*2)
        # self.norm3 = nn.LayerNorm(dims*4)
        # self.norm4 = nn.LayerNorm(dims*8)

        # self.mamba = SS2D_xb(d_model = dims)
        self.mamba1 = SS2D(d_model=dims)
        self.mamba2 = SS2D(d_model=2*dims)
        self.fuse = SS2D_xb(d_model=dims)
        self.dropout1 = nn.Dropout(0.3)
        # self.dropout2 = nn.Dropout(0.2)
        # self.fuse = Mamba(d_model=dims)

    def forward(self, inputs):
        C1 ,C2 = inputs[0],inputs[1]
        C1 = self.mamba1(C1.permute(0,2,3,1)).permute(0,3,1,2)
        C2 = self.mamba2(C2.permute(0,2,3,1)).permute(0,3,1,2)
        # C1 = C1 + self.dropout1(self.mamba1(self.norm1(C1.permute(0,2,3,1))).permute(0,3,1,2))
        # C2 = C2 + self.dropout1(self.mamba2(self.norm2(C2.permute(0,2,3,1))).permute(0,3,1,2))
        B,C,H,W = C1.shape
        F1 = C1.reshape(B,H*W,C)
        F2 = C2.reshape(B,H*W//2,C)
        F = torch.cat((F1,F2),dim=1)
        F = F + self.dropout1(self.fuse(self.norm1(F)))
        # F = self.fuse(F)

        C1 = F[:,:12544,:].reshape(B,C,H,W)
        C2 = F[:,12544:,:].reshape(B,C*2,H//2,W//2)

        return [C1,C2]

class BridgeLayer3_xb(nn.Module):
    def __init__(self, dims=24):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims*2)
        self.norm3 = nn.LayerNorm(dims*4)
        self.norm4 = nn.LayerNorm(dims*8)
        # self.mamba = SS2D_xb(d_model = dims)
        self.mamba1 = SS2D(d_model=dims)
        self.mamba2 = SS2D(d_model=2*dims)
        self.mamba3 = SS2D(d_model=4*dims)
        self.dropout1 = nn.Dropout(0.4)

        self.fuse = SS2D_xb(d_model=dims)

    def forward(self, inputs):
        C1 ,C2,C3 = inputs[0],inputs[1],inputs[2]
        C1 = self.mamba1(C1.permute(0,2,3,1)).permute(0,3,1,2)
        C2 = self.mamba2(C2.permute(0,2,3,1)).permute(0,3,1,2)
        C3 = self.mamba3(C3.permute(0,2,3,1)).permute(0,3,1,2)
        # C1 = C1 + self.dropout1(self.mamba1(self.norm1(C1.permute(0,2,3,1))).permute(0,3,1,2))
        # C2 = C2 + self.dropout1(self.mamba2(self.norm2(C2.permute(0,2,3,1))).permute(0,3,1,2))
        # C3 = C3 + self.dropout1(self.mamba3(self.norm3(C3.permute(0,2,3,1))).permute(0,3,1,2))
        B,C,H,W = C1.shape
        F1 = C1.reshape(B,H*W,C)
        F2 = C2.reshape(B,H*W//2,C)
        F3 = C3.reshape(B,H*W//4,C)
        F = torch.cat((F1,F2,F3),dim=1)
        F = F + self.dropout1(self.fuse(self.norm1(F)))
        # F = self.fuse(F)
        C1 = F[:,:12544,:].reshape(B,C,H,W)
        C2 = F[:,12544:18816,:].reshape(B,C*2,H//2,W//2)
        C3 = F[:,18816:,:].reshape(B,C*4,H//4,W//4)

        return [C1,C2,C3]

class BridgeLayer4_xb(nn.Module):
    def __init__(self, dims=24):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims*2)
        self.norm3 = nn.LayerNorm(dims*4)
        self.norm4 = nn.LayerNorm(dims*8)
        self.mamba1 = SS2D(d_model=dims)
        self.mamba2 = SS2D(d_model=2*dims)
        self.mamba3 = SS2D(d_model=4*dims)
        self.mamba4 = SS2D(d_model=8*dims)
        self.fuse = SS2D_xb(d_model=dims)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, inputs):
        C1 ,C2,C3,C4 = inputs[0],inputs[1],inputs[2],inputs[3]
        C1 = self.mamba1(C1.permute(0,2,3,1)).permute(0,3,1,2)
        C2 = self.mamba2(C2.permute(0,2,3,1)).permute(0,3,1,2)
        C3 = self.mamba3(C3.permute(0,2,3,1)).permute(0,3,1,2)
        C4 = self.mamba4(C4.permute(0,2,3,1)).permute(0,3,1,2)
        # C1 = C1 + self.dropout1(self.mamba1(self.norm1(C1.permute(0,2,3,1))).permute(0,3,1,2))
        # C2 = C2 + self.dropout1(self.mamba2(self.norm2(C2.permute(0,2,3,1))).permute(0,3,1,2))
        # C3 = C3 + self.dropout1(self.mamba3(self.norm3(C3.permute(0,2,3,1))).permute(0,3,1,2))
        # C4 = C4 + self.dropout1(self.mamba4(self.norm4(C4.permute(0,2,3,1))).permute(0,3,1,2))

        B,C,H,W = C1.shape
        F1 = C1.reshape(B,H*W,C)
        F2 = C2.reshape(B,H*W//2,C)
        F3 = C3.reshape(B,H*W//4,C)
        F4 = C4.reshape(B,H*W//8,C)

        F = torch.cat((F1,F2,F3,F4),dim=1)
        F = F + self.dropout1(self.fuse(self.norm1(F)))
        # F = self.fuse(F)
        C1 = F[:,:12544,:].reshape(B,C,H,W)
        C2 = F[:,12544:18816,:].reshape(B,C*2,H//2,W//2)
        C3 = F[:,18816:21952,:].reshape(B,C*4,H//4,W//4)
        C4 = F[:,21952:,:].reshape(B,C*8,H//8,W//8)

        return [C1,C2,C3,C4]


