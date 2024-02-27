#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F

# pointnet from SAL paper: https://github.com/matanatz/SAL/blob/master/code/model/network.py#L14
class PointNet(nn.Module):
    ''' PointNet-based encoder network. Based on: https://github.com/autonomousvision/occupancy_networks
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=256, in_dim=3, hidden_dim=128, **kwargs):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(in_dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        torch.nn.init.constant_(self.fc_mean.weight,0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = self.maxpool
        
        self.head_centroid = nn.Linear(self.c_dim, 3)
        self.head_scale = nn.Linear(self.c_dim, 1)

    def forward(self, p):
        batch_size = p.shape[0]
        # import pdb
        # pdb.set_trace()
        net = self.fc_pos(p.transpose(-1,-2))
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        global_feat = self.fc_mean(self.actvn(net))
        # c_std = self.fc_std(self.actvn(net))
        # import pdb
        # pdb.set_trace()
        scale = self.head_scale(global_feat).squeeze()
        center_pred = self.head_centroid(global_feat)
        z_so3 = torch.ones((batch_size, 256, 3)).cuda().float()
        return center_pred, scale, z_so3, global_feat

    def maxpool(self, x, dim=-1, keepdim=False):
        out, _ = x.max(dim=dim, keepdim=keepdim)
        return out
