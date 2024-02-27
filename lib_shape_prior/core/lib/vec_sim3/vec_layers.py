import torch
from torch import nn
import torch.nn.functional as F
import math
import logging

r"""
The shape convention should be

B,Channel,3,...

"""


def safe_divide(x, y, eps=1e-8):
    # perform x/y and add EPS only on y elements that are near zero
    unstable_mask = (abs(y) < eps).type(y.dtype) * y.sign()
    if (unstable_mask > 0).any():
        logging.debug("safe divide protect!")
    z = x / (y + unstable_mask * eps)
    return z


def channel_equi_vec_normalize(x):
    # B,C,3,...
    assert x.ndim >= 3, "x shape [B,C,3,...]"
    x_dir = F.normalize(x, dim=2)
    x_norm = x.norm(dim=2, keepdim=True)
    x_normalized_norm = F.normalize(x_norm, dim=1)  # normalize across C
    y = x_dir * x_normalized_norm
    return y


class VecLinear(nn.Module):
    r"""
    from pytorch Linear
    Can be SO3 or SE3
    Can have hybrid feature
    The input scalar feature must be invariant
    valid mode: V,h->V,h; V,h->V; V->V,h; V->V; V,h->h
    """

    v_in: int
    v_out: int
    s_in: int
    s_out: int
    weight: torch.Tensor

    def __init__(
        self,
        v_in: int,
        v_out: int,
        s_in=0,
        s_out=0,
        s2v_normalized_scale=True,
        mode="se3",
        device=None,
        dtype=None,
        vs_dir_learnable=True,
        cross=False,
        hyper=False,
    ) -> None:
        mode = mode.lower()
        assert mode in ["so3", "se3"], "mode must be so3 or se3"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.v_in = v_in
        self.v_out = v_out
        self.s_in = s_in
        self.s_out = s_out

        self.hyper_flag = hyper

        assert self.s_out + self.v_out > 0, "vec, scalar output both zero"

        self.se3_flag = mode == "se3"
        if self.se3_flag:
            assert v_in > 1, "se3 layers must have at least two input layers"

        if self.v_out > 0:
            self.weight = nn.Parameter(
                torch.empty(
                    (v_out, v_in - 1 if self.se3_flag else v_in), **factory_kwargs
                )  # if use se3 mode, should constrain the weight to have sum 1.0
            )  # This is the main weight of the vector, due to historical reason, for old checkpoint, not rename this
            self.reset_parameters()

        if (
            self.s_in > 0 and self.v_out > 0
        ):  # if has scalar input, must have a path to fuse to vector
            if self.hyper_flag:
                self.sv_linear = nn.Linear(s_in, int((v_out // 9) * 9))
            else:
                self.sv_linear = nn.Linear(s_in, v_out)
            self.s2v_normalized_scale_flag = s2v_normalized_scale

        if self.s_out > 0:  # if has scalar output, must has vector to scalar path
            self.vs_dir_learnable = vs_dir_learnable
            assert (
                self.vs_dir_learnable
            ), "because non-learnable is not stable numerically, not allowed now"
            if self.vs_dir_learnable:
                self.vs_dir_linear = VecLinear(v_in, v_in, mode="so3")  # TODO: can just have 1 dir
            self.vs_linear = nn.Linear(v_in, s_out)
        if self.s_in > 0 and self.s_out > 0:  # when have s in and s out, has ss path
            self.ss_linear = nn.Linear(s_in, s_out)

        self.cross_flag = cross
        if self.v_out > 0 and self.cross_flag:
            self.v_out_cross = VecLinear(v_in, v_out, mode=mode, cross=False)
            self.v_out_cross_fc = VecLinear(v_out * 2, v_out, mode=mode, cross=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # ! warning, now the initialization will bias to the last channel with larger weight, need better init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.se3_flag:
            self.weight.data += 1.0 / self.v_in

    def forward(self, v_input: torch.Tensor, s_input=None):
        # B,C,3,...; B,C,...

        # First do Vector path if output vector
        v_shape = v_input.shape
        assert v_shape[2] == 3, "not vector neuron"
        if self.v_out > 0:
            if self.se3_flag:
                W = torch.cat(
                    [self.weight, 1.0 - self.weight.sum(-1, keepdim=True)], -1
                ).contiguous()
            else:
                W = self.weight
            v_output = F.linear(v_input.transpose(1, -1), W).transpose(-1, 1)  # B,C,3,...
        else:
            v_output = None

        # Optional Scalar path
        if self.s_in > 0:
            assert s_input is not None, "missing scalar input"
            s_shape = s_input.shape
            assert v_shape[3:] == s_shape[2:]
            # must do scalar to vector fusion
            if self.v_out > 0:
                if self.hyper_flag:
                    raise NotImplementedError()
                    s2v_W = self.sv_linear(s_input.transpose(1, -1)).transpose(-1, 1)
                    B, _, N = s2v_W.shape
                    s2v_W = s2v_W.reshape(B, -1, 3, 3, N)
                    head_K = s2v_W.shape[1]
                    s2v_W = (
                        s2v_W.unsqueeze(1)
                        .expand(-1, int(np.ceil(self.v_out / head_K)), -1, -1, -1, -1)
                        .reshape(B, -1, 3, 3, N)
                    )[:, : self.v_out]
                    s2v_W = s2v_W + torch.eye(3).to(s2v_W.device)[None, None, :, :, None]
                    if self.se3_flag:  # need to scale the rotation part, exclude the center
                        v_new_mean = v_output.mean(dim=1, keepdim=True)
                        v_output = (
                            torch.einsum("bcjn, bcjin->bcin", v_output - v_new_mean, s2v_W)
                            + v_new_mean
                        )
                    else:
                        v_output = torch.einsum("bcjn, bcjin->bcin", v_output, s2v_W)
                else:
                    s2v_invariant_scale = self.sv_linear(s_input.transpose(1, -1)).transpose(-1, 1)
                    if self.s2v_normalized_scale_flag:
                        s2v_invariant_scale = F.normalize(s2v_invariant_scale, dim=1)
                    if self.se3_flag:  # need to scale the rotation part, exclude the center
                        v_new_mean = v_output.mean(dim=1, keepdim=True)
                        v_output = (v_output - v_new_mean) * s2v_invariant_scale.unsqueeze(
                            2
                        ) + v_new_mean
                    else:
                        v_output = v_output * s2v_invariant_scale.unsqueeze(2)
                    # now v_new done

        if self.v_out > 0 and self.cross_flag:
            # do cross production
            v_out_dual = self.v_out_cross(v_input)
            if self.se3_flag:
                v_out_dual_o = v_out_dual.mean(dim=1, keepdim=True)
                v_output_o = v_output.mean(dim=1, keepdim=True)
                v_cross = torch.cross(
                    channel_equi_vec_normalize(v_out_dual - v_out_dual_o),
                    v_output - v_output_o,
                    dim=2,
                )
            else:
                v_cross = torch.cross(channel_equi_vec_normalize(v_out_dual), v_output, dim=2)
            v_cross = v_cross + v_output
            v_output = self.v_out_cross_fc(torch.cat([v_cross, v_output], dim=1))

        if self.s_out > 0:
            # must have the vector to scalar path
            v_sR = v_input - v_input.mean(dim=1, keepdim=True) if self.se3_flag else v_input
            if self.vs_dir_learnable:
                v_sR_dual_dir = F.normalize(self.vs_dir_linear(v_sR), dim=2)
            else:
                v_sR_dual_dir = F.normalize(v_sR.mean(dim=1, keepdim=True), dim=2)
            s_from_v = F.normalize((v_sR * v_sR_dual_dir).sum(dim=2), dim=1)  # B,C,...
            s_from_v = self.vs_linear(s_from_v.transpose(-1, 1)).transpose(-1, 1)
            if self.s_in > 0:
                s_from_s = self.ss_linear(s_input.transpose(-1, 1)).transpose(-1, 1)
                s_output = s_from_s + s_from_v
            else:
                s_output = s_from_v
            return v_output, s_output
        else:
            return v_output


class VecActivation(nn.Module):
    # Also integrate a batch normalization before the actual activation
    # Order: 1.) centered [opt] 2.) normalization in norm [opt] 3.) act 4.) add center [opt]
    def __init__(
        self,
        in_features,
        act_func,
        shared_nonlinearity=False,
        mode="se3",
        normalization=None,
        cross=False,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        assert mode in ["so3", "se3"], "mode must be so3 or se3"
        self.se3_flag = mode == "se3"
        self.shared_nonlinearity_flag = shared_nonlinearity
        self.act_func = act_func

        nonlinear_out = 1 if self.shared_nonlinearity_flag else in_features
        self.lin_dir = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross)
        if self.se3_flag:
            self.lin_ori = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross)
        self.normalization = normalization
        if self.normalization is not None:
            logging.warning("Warning! Set Batchnorm True, not Scale Equivariant")

    def forward(self, x):
        # B,C,3,...
        # warning, there won't be shape check before send to passed in normalization in side this layer
        assert x.shape[2] == 3, "not vector neuron"
        q = x
        k = self.lin_dir(x)
        if self.se3_flag:
            o = self.lin_ori(x)
            q = q - o
            k = k - o

        # normalization if set
        if self.normalization is not None:
            q_dir = F.normalize(q, dim=2)
            q_len = q.norm(dim=2)  # ! note: the shape into BN is [B,C,...]
            # ! Warning! when set the normalization, not scale equivariant!
            q_len_normalized = self.normalization(q_len)
            q = q_dir * q_len_normalized.unsqueeze(2)

        # actual non-linearity on the parallel component length
        k_dir = F.normalize(k, dim=2)
        q_para_len = (q * k_dir).sum(dim=2, keepdim=True)
        q_orthogonal = q - q_para_len * k_dir
        acted_len = self.act_func(q_para_len)
        q_acted = q_orthogonal + k_dir * acted_len
        if self.se3_flag:
            q_acted = q_acted + o
        return q_acted


class VecMeanPool(nn.Module):
    def __init__(self, pooling_dim=-1, **kwargs) -> None:  # Have dummy args here
        super().__init__()
        self.pooling_dim = pooling_dim

    def forward(self, x, return_weight=False):
        if return_weight:
            return x.mean(self.pooling_dim), None
        else:
            return x.mean(self.pooling_dim)


class VecMaxPool(nn.Module):
    def __init__(
        self,
        in_features,
        shared_nonlinearity=False,
        mode="se3",
        pooling_dim=-1,
        softmax_factor=-1.0,  # if positive, use softmax
        k_prediction="lin",  # when setting to mean and use soft max, has attention
        attention_k_blk=True,  # if set, the mean feature will first processed by a key block
        softmax_norm_compression="sigmoid",  # for compression before inner product to stablize
        cross=False,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        assert mode in ["so3", "se3"], "mode must be so3 or se3"
        self.se3_flag = mode == "se3"
        self.shared_nonlinearity_flag = shared_nonlinearity

        nonlinear_out = 1 if self.shared_nonlinearity_flag else in_features
        self.k_prediction = k_prediction
        if self.k_prediction == "lin":
            self.lin_dir = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross)
        self.use_attention_k_blk = attention_k_blk
        if self.use_attention_k_blk:
            self.attention_blk = VecResBlock(
                in_features,
                in_features,
                in_features,
                mode=mode,
                act_func=nn.LeakyReLU(negative_slope=0.2),
                last_activate=False,
                cross=cross,
            )
        if self.se3_flag:
            self.lin_ori = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross)

        assert (
            pooling_dim < 0 or pooling_dim >= 3
        ), "invalid pooling dim, the input should have [B,C,3,...] shape"
        self.pooling_dim = pooling_dim

        self.softmax_factor = softmax_factor
        self.softmax_norm_compression = softmax_norm_compression
        self.softmax_compression_method = {"sigmoid": self.sigmoid_norm, "exp": self.exp_norm}[
            self.softmax_norm_compression
        ]

    def sigmoid_norm(self, x, dim=2):
        len = x.norm(dim=dim, keepdim=True)
        dir = F.normalize(x, dim=dim)
        y = dir * torch.sigmoid(len)
        return y

    def exp_norm(self, x, dim=2):
        len = x.norm(dim=dim, keepdim=True)
        dir = F.normalize(x, dim=dim)
        compressed_len = 1.0 - torch.exp(-len)
        y = dir * compressed_len
        return y

    def forward(self, x, return_weight=False):
        # B,C,3,...
        assert x.shape[2] == 3, "not vector neuron"
        q = x
        # get k
        if self.k_prediction == "lin":
            k = self.lin_dir(x)
        elif self.k_prediction == "mean":  # Attention!
            k = x.mean(dim=self.pooling_dim, keepdim=True)
            if self.use_attention_k_blk:
                k = self.attention_blk(k)
        else:
            raise NotImplementedError()
        if self.se3_flag:
            o = self.lin_ori(x)
            q = q - o
            k = k - o
        k_scale = k.mean(dim=1, keepdim=True).norm(dim=2, keepdim=True)
        k = k.expand_as(x)
        k_scale_inv = self.softmax_compression_method(safe_divide(k, k_scale), dim=2)

        if self.softmax_factor > 0.0:
            # q_scale = q.mean(dim=1, keepdim=True).norm(dim=2, keepdim=True)
            q_scale_inv = self.softmax_compression_method(
                safe_divide(q, k_scale), dim=2
            )  # ! note, here divide by k_scale, because k is pooled, more stable and not easy to have numerical issue
            sim3_invariant_w = (q_scale_inv * k_scale_inv).mean(dim=2, keepdim=True)
            pooling_w = torch.softmax(self.softmax_factor * sim3_invariant_w, dim=self.pooling_dim)
            out = (x * pooling_w).sum(self.pooling_dim)
            if return_weight:
                return out, pooling_w
            else:
                return out
        else:  # hard max pool
            q_para_len = (q * k_scale_inv).sum(dim=2, keepdim=True)
            selection = torch.argmax(q_para_len, dim=self.pooling_dim, keepdim=True)
            _expand_args = [-1] * selection.ndim
            _expand_args[2] = 3
            selection = selection.expand(*_expand_args)
            selected_x = torch.gather(input=x, dim=self.pooling_dim, index=selection)
            selected_x = selected_x.squeeze(self.pooling_dim)
            if return_weight:
                return selected_x, None
            else:
                return selected_x


class VecMaxPoolV2(nn.Module):
    # newer version, remove all safe divide, use channel wise normalizaiton to factor out scale
    def __init__(
        self,
        in_features,
        shared_nonlinearity=False,
        mode="se3",
        pooling_dim=-1,
        softmax_factor=-1.0,  # if positive, use softmax
        k_prediction="mean",  # when setting to mean and use soft max, has attention
        attention_k_blk=True,  # if set, the mean feature will first processed by a key block
        softmax_norm_compression="sigmoid",  # for compression before inner product to stablize
        cross=False,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        assert mode in ["so3", "se3"], "mode must be so3 or se3"
        self.se3_flag = mode == "se3"
        self.shared_nonlinearity_flag = shared_nonlinearity

        nonlinear_out = 1 if self.shared_nonlinearity_flag else in_features
        assert k_prediction == "mean", "v2 only support mean"
        self.use_attention_k_blk = attention_k_blk
        if self.use_attention_k_blk:
            self.attention_blk = VecResBlock(
                in_features,
                in_features,
                in_features,
                mode=mode,
                act_func=nn.LeakyReLU(negative_slope=0.2),
                last_activate=False,
                cross=cross,
            )
        if self.se3_flag:
            self.lin_ori = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross)

        assert (
            pooling_dim < 0 or pooling_dim >= 3
        ), "invalid pooling dim, the input should have [B,C,3,...] shape"
        self.pooling_dim = pooling_dim

        self.softmax_factor = softmax_factor

    def forward(self, x, return_weight=False):
        # B,C,3,...
        reshape_flag = False
        if x.ndim == 5:  # B,C,3,N,K
            B, C, _, N, K = x.shape
            x = x.permute(0, 3, 1, 2, 4).reshape(B * N, C, 3, K)
            reshape_flag = True
        else:
            B, C, _, N = x.shape

        assert not return_weight or x.ndim == 4, "now don't support 5dim return weight"
        assert x.shape[2] == 3, "not vector neuron"

        q = x
        # get k
        k = x.mean(dim=self.pooling_dim, keepdim=True)
        if self.use_attention_k_blk:
            # todo: this may not working for the 5 dim input for now
            k = self.attention_blk(k)

        if self.se3_flag:
            o = self.lin_ori(k)
            q = q - o
            k = k - o
        k_scale_inv = channel_equi_vec_normalize(k)
        if self.softmax_factor > 0.0:
            q_scale_inv = channel_equi_vec_normalize(q)
            sim3_invariant_w = (q_scale_inv * k_scale_inv).mean(dim=2, keepdim=True)
            pooling_w = torch.softmax(self.softmax_factor * sim3_invariant_w, dim=self.pooling_dim)
            out = (x * pooling_w).sum(self.pooling_dim)
            if reshape_flag:
                out = out.reshape(B, N, C, 3).permute(0, 2, 3, 1)
            if return_weight:
                return out, pooling_w
            else:
                return out
        else:  # hard max pool
            q_para_len = (q * k_scale_inv).sum(dim=2, keepdim=True)
            selection = torch.argmax(q_para_len, dim=self.pooling_dim, keepdim=True)
            _expand_args = [-1] * selection.ndim
            _expand_args[2] = 3
            selection = selection.expand(*_expand_args)
            selected_x = torch.gather(input=x, dim=self.pooling_dim, index=selection)
            selected_x = selected_x.squeeze(self.pooling_dim)
            if reshape_flag:
                selected_x = selected_x.reshape(B, N, C, 3).permute(0, 2, 3, 1)
            if return_weight:
                return selected_x, None
            else:
                return selected_x


class VecLinearNormalizeActivate(nn.Module):
    # support vector scalar hybrid operation
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_func,
        s_in_features=0,
        s_out_features=0,
        shared_nonlinearity=False,
        normalization=None,
        mode="se3",
        s_normalization=None,
        vs_dir_learnable=True,
        cross=False,
    ) -> None:
        super().__init__()

        self.scalar_out_flag = s_out_features > 0
        self.lin = VecLinear(
            in_features,
            out_features,
            s_in_features,
            s_out_features,
            mode=mode,
            vs_dir_learnable=vs_dir_learnable,
            cross=cross,
        )
        self.act = VecActivation(
            out_features, act_func, shared_nonlinearity, mode, normalization, cross=cross
        )
        self.s_normalization = s_normalization
        self.act_func = act_func
        return

    def forward(self, v, s=None):
        if self.scalar_out_flag:  # hybrid mode
            v_out, s_out = self.lin(v, s)
            v_act = self.act(v_out)
            if self.s_normalization is not None:
                s_out = self.s_normalization(s_out)
            s_act = self.act_func(s_out)
            return v_act, s_act
        else:
            v_out = self.lin(v, s)
            v_act = self.act(v_out)
            return v_act


class VecResBlock(nn.Module):
    # ! warning, here different from the original vnn code, the order changed, first linear and the activate, so the last layer has an act option; Note, the network will be different especially when applying max pool, in vnn original code, first do pooling and then do the activation, but here we first do activation and then do the pooling
    # * if set scalar out channels, return 2 values, else return 1 values, not elegant, but this is for running with old codes and models
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        act_func,
        mode="se3",
        s_in_features=0,
        s_out_features=0,
        s_hidden_features=0,
        use_bn=False,
        s_use_bn=False,
        last_activate=True,
        vs_dir_learnable=True,
        cross=False,
    ) -> None:
        super().__init__()
        if use_bn:
            logging.warning("Warning! Set Batchnorm True, not Scale Equivariant")

        self.last_activate = last_activate
        self.act_func = act_func

        self.s_in_features = s_in_features
        self.s_out_features = s_out_features
        self.s_hidden_features = s_hidden_features
        self.s_use_bn = s_use_bn

        self.fc0 = VecLinearNormalizeActivate(
            in_features=in_features,
            out_features=hidden_features,
            s_in_features=s_in_features,
            s_out_features=s_hidden_features,
            act_func=act_func,
            shared_nonlinearity=False,
            mode=mode,
            normalization=nn.BatchNorm1d(hidden_features) if use_bn else None,
            s_normalization=nn.BatchNorm1d(s_hidden_features)
            if s_use_bn and s_hidden_features > 0
            else None,
            vs_dir_learnable=vs_dir_learnable,
            cross=cross,
        )

        self.lin1 = VecLinear(
            v_in=hidden_features,
            v_out=out_features,
            s_in=s_hidden_features,
            s_out=s_out_features,
            mode=mode,
            vs_dir_learnable=vs_dir_learnable,
            cross=cross,
        )
        if s_out_features > 0 and s_use_bn and self.last_activate:
            self.s_bn1 = nn.BatchNorm1d(s_out_features)

        if self.last_activate:
            self.act2 = VecActivation(
                in_features=out_features,
                act_func=act_func,
                shared_nonlinearity=False,
                mode=mode,
                normalization=nn.BatchNorm1d(out_features) if use_bn else None,
                cross=cross,
            )

        self.shortcut = (
            None if in_features == out_features else VecLinear(in_features, out_features, mode=mode)
        )
        if (
            self.s_in_features > 0
            and self.s_out_features > 0
            and self.s_in_features != self.s_out_features
        ):
            self.s_shortcut = nn.Linear(self.s_in_features, self.s_out_features, bias=True)
        else:
            self.s_shortcut = None

        self.se3_flag = mode == "se3"
        if self.se3_flag:
            # ! this is because the short cut add another t!
            self.subtract = VecLinear(in_features, out_features, mode="se3")

    def sv_wrapper(self, module, vec, scalar):
        # to work with old interface, can not break the vecnormact return values, so add a wrapper
        out = module(vec, scalar)
        if isinstance(out, tuple):
            return out[0], out[1]
        else:
            return out, None

    def forward(self, v, s=None):
        # strict shape x: [B,C,3,N]; [B,C,N]
        assert (
            v.ndim == 4
        ), "Residual block only supports input dim = 4"  # this design is for the case using BN, the BN should specify the shape
        assert v.shape[2] == 3, "vec dim should be at dim [2]"
        if self.s_in_features == 0:
            s = None  # for more flexible usage, the behavior is determined by init, not passed in args

        v_net, s_net = self.sv_wrapper(self.fc0, v, s)
        dv, ds = self.sv_wrapper(self.lin1, v_net, s_net)

        if self.shortcut is not None:
            v_s = self.shortcut(v)
        else:
            v_s = v
        v_out = v_s + dv
        if self.se3_flag:
            v_out = v_out - self.subtract(v)
        if self.last_activate:
            v_out = self.act2(v_out)

        if self.s_shortcut is not None:
            assert s is not None and ds is not None
            s_s = self.s_shortcut(s.transpose(-1, 1)).transpose(-1, 1)
            s_out = s_s + ds
        elif ds is not None:  # s_in == s_out or s_in = 0
            if s is None:
                s_out = ds
            else:
                s_out = s + ds
        else:
            s_out = None

        if s_out is not None:
            if self.last_activate:
                if self.s_use_bn:
                    s_out = self.s_bn1(s_out)
                s_out = self.act_func(s_out)
            return v_out, s_out
        else:
            return v_out


def augment(s, R, t, x):
    # s: [B]; R: [B,3,3]; t: [B,3,1]
    # DEBUG function
    B = x.shape[0]
    assert s.ndim == 1 and R.ndim == 3 and t.ndim == 3
    assert s.shape[0] == R.shape[0] == t.shape[0] == B
    if x.ndim == 4:
        B, C, _, N = x.shape
        aug_x = torch.einsum("bij,bcjn->bcin", R, s[:, None, None, None] * x) + t[:, None, ...]
        # aug_x = x * s[:, None, None, None]  # B,C,3,N
        # aug_x = aug_x.permute(0, 1, 3, 2)[..., None]  # B,C,N,3,1
        # aug_x = torch.matmul(R[:, None, None, :, :].expand(-1, C, N, -1, -1), aug_x)
        # aug_x = aug_x + t[:, None, None, :, :]
        # aug_x = aug_x.squeeze(-1)
        # aug_x = aug_x.permute(0, 1, 3, 2)
    elif x.ndim == 3:
        B, C, _ = x.shape
        aug_x = torch.einsum("bij,bcj->bci", R, s[:, None, None] * x) + t.transpose(2, 1)
    else:
        raise NotImplementedError()
    return aug_x


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    import numpy as np
    import random

    np.random.seed(0)
    random.seed(0)
    # # debug
    device = torch.device("cuda")
    B, N = 5, 1024

    # for mode in ["se3", "so3"]:
    #     lin = VecLinear(in_features=64, out_features=128, mode=mode).to(device)
    #     act = VecActivation(
    #         in_features=128,
    #         act_func=nn.LeakyReLU(negative_slope=0.2),
    #         mode=mode,
    #         normalization=nn.BatchNorm1d(128),
    #     ).to(device)
    #     block = VecResBlock(
    #         128,
    #         256,
    #         hidden_features=128,
    #         act_func=nn.LeakyReLU(negative_slope=0.2),
    #         mode=mode,
    #         use_bn=True,
    #         last_activate=True,
    #     ).to(device)
    #     pooling_layer = VecMaxPool(in_features=256, mode=mode, softmax_factor=1.0).to(device)
    #     x = torch.rand(5, 64, 3, 1024).to(device)
    #     y = lin(x)
    #     z = act(y)
    #     u = block(z)
    #     w = pooling_layer(u)
    #     print(x.shape)
    #     print(y.shape)
    #     print(z.shape)
    #     print(u.shape)
    #     print(w.shape)

    # debug equivariance
    torch.set_default_dtype(torch.float64)
    x = torch.rand(B, 128, 3, N).to(device)
    mode = "se3"

    # net = VecResBlock(
    #     in_features=128,
    #     out_features=129,
    #     hidden_features=127,
    #     s_in_features=64,
    #     s_out_features=0,
    #     s_hidden_features=0,
    #     act_func=nn.LeakyReLU(negative_slope=0.2),
    #     mode=mode,
    #     use_bn=False,
    #     s_use_bn=True,
    #     last_activate=True,
    # ).to(device)
    # net.eval()

    # net = VecResBlock(
    #     128,
    #     256,
    #     hidden_features=128,
    #     act_func=nn.LeakyReLU(negative_slope=0.2),
    #     mode=mode,
    #     use_bn=False,
    #     last_activate=True,
    #     cross=True,
    # ).to(device)
    # net = VecLinear(in_features=128, out_features=256, mode=mode).to(device)
    # net = VecLinearNormalizeActivate(
    #     128, 256, act_func=nn.LeakyReLU(negative_slope=0.2),
    # )
    # net = VecActivation(
    #     in_features=128,
    #     act_func=nn.LeakyReLU(negative_slope=0.2),
    #     mode=mode,
    #     normalization=None,
    # ).to(device)
    # net = VecMaxPool(
    #     in_features=128,
    #     mode="se3",
    #     softmax_factor=1.0,
    #     softmax_norm_compression="exp",
    #     k_prediction="mean",
    # ).to(device)
    # net = VecMaxPoolV2(
    #     in_features=128,
    #     mode="se3",
    #     softmax_factor=1.0,
    #     k_prediction="mean",
    # ).to(device)

    net = VecLinear(v_in=128, v_out=64, s_in=64, s_out=32, mode=mode, cross=False, hyper=True).to(
        device
    )

    scalar_feat_in = torch.rand(B, 64, N).to(device)
    # v = net(x, scalar_feat_in)
    v, scalar_feat_out = net(x, scalar_feat_in)
    # v = net(x)

    for _ in range(10):
        # t = torch.rand(B, 3, 1).to(device)
        t = torch.zeros(B, 3, 1).to(device)
        R = [torch.from_numpy(Rotation.random().as_matrix()) for _ in range(B)]
        R = torch.stack(R, 0).to(device).type(x.dtype)
        s = torch.rand(B).to(device)
        # s = torch.ones(B).to(device)

        aug_v = augment(s, R, t, v)
        aug_x = augment(s, R, t, x)

        aug_v_hat, scalar_feat_out_hat = net(aug_x, scalar_feat_in)
        # aug_v_hat = net(aug_x, scalar_feat_in)
        # aug_v_hat = net(aug_x)

        v_error = abs(aug_v - aug_v_hat)
        print("v", v_error.max())
        # s_error = abs(scalar_feat_out - scalar_feat_out_hat)
        # print("s", s_error.max())
        print("-" * 20)
