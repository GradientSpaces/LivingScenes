# From VN DGCNN add scale equiv, no se(3) is considered in plain dgcnn

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
import logging

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vec_layers import *
from vec_layers import VecLinearNormalizeActivate as VecLNA


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VecDGCNN(nn.Module):
    # ! use mean pooling
    def __init__(
        self,
        hidden_dim=128,
        c_dim=128,
        first_layer_knn=16,
        scale_factor=640.0,
        leak_neg_slope=0.2,
        use_dg=False,  # if true, every layer use a new knn topology
    ):
        super().__init__()

        self.use_dg = use_dg
        if self.use_dg:
            logging.info("DGCNN use Dynamic Graph (different from the input topology)")
        self.scale_factor = scale_factor

        # * prepare layers
        self.h_dim = hidden_dim
        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)

        self.c_dim = c_dim
        self.k = first_layer_knn

        self.conv1 = VecLNA(2, hidden_dim, mode="so3", act_func=act_func)
        self.conv2 = VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)
        self.conv3 = VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)
        self.conv4 = VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)

        self.pool1 = meanpool
        self.pool2 = meanpool
        self.pool3 = meanpool
        self.pool4 = meanpool

        self.conv_c = VecLNA(
            hidden_dim * 4, c_dim, mode="so3", act_func=act_func, shared_nonlinearity=True
        )

        self.fc_inv = VecLinear(c_dim, c_dim, mode="so3")

    def get_graph_feature(self, x: torch.Tensor, k: int, knn_idx=None):
        # x: B,C,3,N return B,C*2,3,N,K

        B, C, _, N = x.shape
        if knn_idx is None:
            # if knn_idx is not none, compute the knn by x distance; ndf use fixed knn as input topo
            _x = x.reshape(B, -1, N)
            _, knn_idx, neighbors = knn_points(
                _x.transpose(2, 1), _x.transpose(2, 1), K=k, return_nn=True
            )  # B,N,K; B,N,K; B,N,K,D
            neighbors = neighbors.reshape(B, N, k, C, 3).permute(0, -2, -1, 1, 2)
        else:  # gather from the input knn idx
            assert knn_idx.shape[-1] == k, f"input knn gather idx should have k={k}"
            neighbors = torch.gather(
                x[..., None, :].expand(-1, -1, -1, N, -1),
                dim=-1,
                index=knn_idx[:, None, None, ...].expand(-1, C, 3, -1, -1),
            )  # B,C,3,N,K
        x = x[..., None].expand_as(neighbors)
        y = torch.cat([neighbors - x, x], 1)
        return y, knn_idx  # B,C*2,3,N,K

    def forward(self, x):

        B, _, N = x.shape
        x = x.unsqueeze(1)
        x, knn_idx = self.get_graph_feature(x, k=self.k, knn_idx=None)
        x = self.conv1(x)
        x1 = self.pool1(x)
        if self.use_dg:
            knn_idx = None

        x, _ = self.get_graph_feature(x1, k=self.k, knn_idx=knn_idx)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x, _ = self.get_graph_feature(x2, k=self.k, knn_idx=knn_idx)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x, _ = self.get_graph_feature(x3, k=self.k, knn_idx=knn_idx)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_c(x)
        x = x.mean(dim=-1, keepdim=False)

        z_so3 = channel_equi_vec_normalize(x)  # without scale
        scale = x.norm(dim=-1).mean(1) * self.scale_factor
        z_inv_dual = self.fc_inv(x[..., None]).squeeze(-1)
        v_inv = (channel_equi_vec_normalize(z_inv_dual) * z_so3).sum(-1)
        out_inv_feat = v_inv

        return scale, z_so3, out_inv_feat


class VecDGCNN_v2(nn.Module):
    # ! use mean pooling
    def __init__(
        self,
        hidden_dim=128,
        c_dim=128,
        num_layers=4,
        use_input_cross_feature=True,
        use_res_global_conv=True,
        res_global_start_layer=0,
        first_layer_knn=16,
        scale_factor=640.0,
        leak_neg_slope=0.2,
        use_dg=False,  # if true, every layer use a new knn topology
    ):
        super().__init__()

        self.use_dg = use_dg
        if self.use_dg:
            logging.info("DGCNN use Dynamic Graph (different from the input topology)")
        self.scale_factor = scale_factor

        self.use_input_cross_feature = use_input_cross_feature  # use 3 channels instead of 2
        self.use_res_global_conv = use_res_global_conv  # use a global point net as well
        self.res_global_start_layer = res_global_start_layer
        self.num_layers = num_layers

        self.h_dim = hidden_dim
        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)

        self.c_dim = c_dim
        self.k = first_layer_knn
        self.pool = meanpool

        self.global_conv_list, self.conv_list = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.conv_list.append(
                    VecLNA(
                        3 if use_input_cross_feature else 2,
                        hidden_dim,
                        mode="so3",
                        act_func=act_func,
                    )
                )
            else:
                self.conv_list.append(
                    VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)
                )
            if use_res_global_conv and i >= self.res_global_start_layer:
                self.global_conv_list.append(
                    VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)
                )

        self.conv_c = VecLNA(
            hidden_dim * self.num_layers,
            c_dim,
            mode="so3",
            act_func=act_func,
            shared_nonlinearity=True,
        )

        self.fc_inv = VecLinear(c_dim, c_dim, mode="so3")

    def get_graph_feature(self, x: torch.Tensor, k: int, knn_idx=None, cross=False):
        # x: B,C,3,N return B,C*2,3,N,K

        B, C, _, N = x.shape
        if knn_idx is None:
            # if knn_idx is not none, compute the knn by x distance; ndf use fixed knn as input topo
            _x = x.reshape(B, -1, N)
            _, knn_idx, neighbors = knn_points(
                _x.transpose(2, 1), _x.transpose(2, 1), K=k, return_nn=True
            )  # B,N,K; B,N,K; B,N,K,D
            neighbors = neighbors.reshape(B, N, k, C, 3).permute(0, -2, -1, 1, 2)
        else:  # gather from the input knn idx
            assert knn_idx.shape[-1] == k, f"input knn gather idx should have k={k}"
            neighbors = torch.gather(
                x[..., None, :].expand(-1, -1, -1, N, -1),
                dim=-1,
                index=knn_idx[:, None, None, ...].expand(-1, C, 3, -1, -1),
            )  # B,C,3,N,K
        x_padded = x[..., None].expand_as(neighbors)

        if cross:
            x_dir = F.normalize(x, dim=2)
            x_dir_padded = x_dir[..., None].expand_as(neighbors)
            cross = torch.cross(x_dir_padded, neighbors)
            y = torch.cat([cross, neighbors - x_padded, x_padded], 1)
        else:
            y = torch.cat([neighbors - x_padded, x_padded], 1)
        return y, knn_idx  # B,C*2,3,N,K

    def forward(self, x):

        B, _, N = x.shape
        x = x.unsqueeze(1)

        feat_list = []
        for i in range(self.num_layers):
            if i == 0:  # first layer
                y, knn_idx = self.get_graph_feature(
                    x, k=self.k, knn_idx=None, cross=self.use_input_cross_feature
                )
                if self.use_dg:
                    knn_idx = None  # compute every time
            else:
                y, _ = self.get_graph_feature(x, k=self.k, knn_idx=knn_idx)
            y = self.pool(self.conv_list[i](y))  # get local KNN msg passing out feat
            if self.use_res_global_conv and i >= self.res_global_start_layer:
                global_y = self.pool(y)
                y = torch.cat([y, global_y[..., None].expand_as(y)], 1)
                y = self.global_conv_list[i - self.res_global_start_layer](y)
            feat_list.append(y)
            x = y

        x = torch.cat(feat_list, dim=1)
        x = self.conv_c(x)
        x = x.mean(dim=-1, keepdim=False)

        z_so3 = channel_equi_vec_normalize(x)  # without scale
        scale = x.norm(dim=-1).mean(1) * self.scale_factor
        z_inv_dual = self.fc_inv(x[..., None]).squeeze(-1)
        v_inv = (channel_equi_vec_normalize(z_inv_dual) * z_so3).sum(-1)
        out_inv_feat = v_inv

        return scale, z_so3, out_inv_feat


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    torch.set_default_dtype(torch.float32)

    device = torch.device("cuda")
    B, N = 16, 512
    pcl = torch.rand(B, 3, N).to(device)  # .double()
    net = VecDGCNN_v2(
        hidden_dim=128,
        c_dim=128,
        first_layer_knn=20,
        scale_factor=640.0,
        leak_neg_slope=0.2,
        use_dg=True,
    ).to(device)
    net.eval()

    with torch.no_grad():
        scale, so3_feat, inv_feat = net(pcl)

        for _ in range(10):
            t = torch.rand(B, 3, 1).to(device) - 0.5
            t = t * 0
            R = [torch.from_numpy(Rotation.random().as_matrix()) for _ in range(B)]
            R = torch.stack(R, 0).to(device).type(scale.dtype)
            s = torch.rand(B).to(device)
            # s = torch.ones(B).to(device)
            aug_pcl = torch.einsum("bij,bjn->bin", R.clone(), pcl * s[:, None, None]) + t

            aug_scale_hat, aug_so3_feat_hat, aug_inv_feat_hat = net(aug_pcl)

            aug_scale = scale * s
            aug_so3_feat = torch.einsum("bij,bnj->bni", R.clone(), so3_feat)

            error_so3_feat = torch.einsum("bij,bkj->bik", aug_so3_feat, aug_so3_feat_hat)
            error_so3_feat = (
                torch.acos(
                    torch.clamp(
                        (
                            error_so3_feat[:, 0, 0]
                            + error_so3_feat[:, 1, 1]
                            + error_so3_feat[:, 2, 2]
                            - 1.0
                        )
                        / 2.0
                        - 1.0,
                        1.0,
                    )
                )
                / np.pi
                * 180.0
            )
            error_so3_feat = error_so3_feat.max()

            error_inv_feat = (abs(aug_inv_feat_hat - inv_feat)).max()

            error_scale = abs(aug_scale - aug_scale_hat).max()

            print(f"so3_feat error {error_so3_feat} deg")
            print(f"Inv_feat error {error_inv_feat}")
            print(f"Scale error {error_scale}")
            print("-" * 20)
