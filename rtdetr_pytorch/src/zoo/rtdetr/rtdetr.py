"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        n_parameters = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print("No of Parameters in backbone: ", n_parameters/1e6)

        self.decoder = decoder
        n_parameters = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print("No of Parameters in decoder: ", n_parameters/1e6)

        self.encoder = encoder
        n_parameters = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print("No of Parameters in encoder: ",n_parameters/1e6)

        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
