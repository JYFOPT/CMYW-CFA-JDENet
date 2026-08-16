#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:56:48 2021

@author: xingwenzhu
"""

import torch
import torch.nn as nn
# import nets.essa_basicblock as B
import nets.Rest_basicblock as B
        
class RESTCANet(nn.Module):
    '''RSTCANet'''
    def __init__(self, in_nc=1, out_nc=3, patch_size=2, nc=72, window_size=6, img_size=[64, 64], num_heads=6,
                 depths = 2):
        super(RESTCANet, self).__init__()
        
        # m_pp = B.PixelUnShuffle(patch_size)
        # m_le = B.LinearEmbedding(in_channels=in_nc*patch_size*patch_size, out_channels=nc)
        pos_drop = nn.Dropout(p=0.)
        # self.head = B.sequential(m_pp, m_le, pos_drop)
        m_pp = B.PixelUnShuffle(patch_size)
        self.head = B.conv(in_nc, nc, bias=True, mode='C')
        
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]

        
        self.m_body = B.Body(patches_resolution,
                             depths=depths,
                             num_heads=num_heads,
                             nc=nc,
                             window_size=window_size)

        # self.m_ipp = B.PatchUnEmbedding(nc)
        # self.m_conv = B.conv(nc//patch_size//patch_size, nc, bias=True, mode='C')
        self.m_final_conv = B.conv(nc, out_nc, bias=True, mode='C')

    def forward(self, x0):
        
        '''
        encoder
        '''
        # shallow feature extraction
        x1 = self.head(x0)

        #deep feature extraction
        # x_size = (x0.shape[2]//2, x0.shape[3]//2)  #248,248

        x = self.m_body(x1)

        # x = self.m_ipp(x, x_size)
        # x = self.m_conv(x)

        #image reconstruction
        x = self.m_final_conv(x)
        
        return x
