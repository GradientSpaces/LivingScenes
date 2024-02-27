import torch
import torch.nn as nn

"""
modified from: https://github.com/qinglew/PCN-PyTorch/blob/master/models/pcn.py
and
https://openaccess.thecvf.com/content/WACV2022/supplemental/Duggal_Mending_Neural_Implicit_WACV_2022_supplemental.pdf
"""
class PCNet(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, latent_dim=1024, output_dim=256, **kwargs):
        super().__init__()

        
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Tanh()
        )

        self.head_centroid = nn.Linear(self.output_dim, 3)
        self.head_scale = nn.Linear(self.output_dim, 1)

    def forward(self, xyz):
        B, _, N = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz)                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        feature_global = self.mlp(feature_global)
        scale = self.head_scale(feature_global).squeeze()
        center_pred = self.head_centroid(feature_global)
        z_so3 = torch.ones((B, 256, 3)).cuda().float()
        return center_pred, scale, z_so3, feature_global
