import sys, os, glob
import os.path as osp

import numpy as np
from datetime import datetime
import torch
from torch import nn
import yaml, random
from torch import distributions as dist
from pytorch3d.ops import sample_farthest_points as fps
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "./lib_shape_prior/")))
from lib_shape_prior.core.lib.implicit_func.onet_decoder import Decoder, DecoderCBatchNorm, DecoderCat
from lib_shape_prior.core.lib.implicit_func.deepsdf_decoder import DeepSDF_Decoder as Deepsdf
from lib_shape_prior.core.lib.vec_sim3.vec_dgcnn import VecDGCNN, VecDGCNN_v2
from lib_shape_prior.core.lib.vec_sim3.vec_dgcnn_atten import VecDGCNN_att
from lib_shape_prior.core.lib.vec_sim3.pcnet import PCNet
from lib_shape_prior.core.models.utils.occnet_utils.mesh_extractor2 import Generator3D as Generator3D_MC

import logging


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_log_dir(path, resume=False):
    if osp.exists(path):
        if resume:
            viz_dir = osp.join(path, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            back_dir = osp.join(path, "backup")
            os.makedirs(back_dir, exist_ok=True)
            return path, viz_dir, back_dir
        else:
            print("warning, old log exists, not resume, move to bck")
            os.makedirs(path + "_bck", exist_ok=True)
            time_stamp = datetime.now().strftime("%H_%M_%S")
            os.system(f"mv {path} {osp.join(path+'_bck', osp.basename(path)+time_stamp)}")
    os.makedirs(path, exist_ok=False)
    viz_dir = osp.join(path, "viz")
    os.makedirs(viz_dir)
    back_dir = osp.join(path, "backup")
    os.makedirs(back_dir)
    return path, viz_dir, back_dir


def cfg_with_default(cfg, key_list, default):
    root = cfg
    for k in key_list:
        if k in root.keys():
            root = root[k]
        else:
            return default
    return root


def count_param(net):
    return sum(param.numel() for param in net.parameters())


def load_models_dict(cfg, device):
    models_name_list = [k for k in cfg["shape_priors"].keys()]
    MODEL_DICT = nn.ModuleDict()
    for name in models_name_list:
        cate_model_config = cfg["shape_priors"][name]
        cate_model_config["working_dir"] = cfg["working_dir"]
        MODEL_DICT[name] = (
            Shape_Prior(
                cate_model_config,
                model_id=name,
                use_double=cfg_with_default(cfg, ["solver_global", "use_double"], True),
            )
            .to(device)
            .eval()
        )
    return MODEL_DICT


class Shape_Prior(nn.Module):
    # a wrapper for shape prior, used only for inference
    def __init__(self, cfg, model_id, use_double=True) -> None:
        super().__init__()
        self.model_id = model_id

        working_dir = cfg["working_dir"]

        with open(osp.join(working_dir, cfg["field_cfg"]), "r") as f:
            self.field_cfg = yaml.full_load(f)

        self.decoder_type = cfg_with_default(
            self.field_cfg, ["model", "decoder_type"], "cbatchnorm"
        )
        decoder_class = {"decoder": Decoder, 
                         "cbatchnorm": DecoderCBatchNorm, 
                         "inner": DecoderCat, 
                         "deepsdf": Deepsdf,
                         "inner_deepsdf": Deepsdf,
                         "inv_mlp": DecoderCat
                         }[self.decoder_type]
        self.encoder_type = cfg_with_default(
            self.field_cfg, ["model", "encoder_type"], "sim3pointres"
        )
        encoder_class = {
            "vecdgcnn": VecDGCNN,
            "vecdgcnn2": VecDGCNN_v2,
            "vecdgcnn_atten": VecDGCNN_att,
            "pcnet": PCNet,
        }[self.encoder_type]

        encoder = encoder_class(**self.field_cfg["model"]["encoder"])
        decoder = decoder_class(**self.field_cfg["model"]["decoder"])
        self.field_input_n = self.field_cfg["dataset"]["n_pcl"]

        f_param = torch.load(osp.join(working_dir, cfg["field_pt"]))
        field_loaded_ep = f_param["epoch"]
        f_param = f_param["model_state_dict"]
        encoder.load_state_dict(
            {".".join(k.split(".")[2:]): f_param[k] for k in f_param.keys() if "encoder" in k},
            strict=True,
        )
        decoder.load_state_dict(
            {".".join(k.split(".")[2:]): f_param[k] for k in f_param.keys() if "decoder" in k},
            strict=True,
        )

        # support cls head
        self.use_cls = cfg_with_default(self.field_cfg, ["model", "use_cls"], False)
        if self.use_cls:
            _c_dim = self.field_cfg["model"]["encoder"]["c_dim"]
            self.cls_head = nn.Sequential(
                nn.Linear(_c_dim, _c_dim),
                nn.Sigmoid(),
                nn.Linear(_c_dim, _c_dim),
                nn.Sigmoid(),
                nn.Linear(_c_dim, self.field_cfg["model"]["num_cates"]),
            )
            self.cls_head.load_state_dict(
                {".".join(k.split(".")[2:]): f_param[k] for k in f_param.keys() if "cls_head" in k},
                strict=True,
            )
        else:
            self.cls_head = None

        self.use_double = use_double
        if use_double:
            self.encoder = encoder.double()
        else:
            self.encoder = encoder
        self.decoder = FieldWrapper(
            decoder,
            sdf2occ_factor=cfg_with_default(self.field_cfg, ["model", "sdf2occ_factor"], -1.0),
            decoder_type=self.decoder_type,
        )

        logging.info(f"Model {self.model_id} successfully loaded at epoch {field_loaded_ep}.")
        logging.info(f"Encoder with {count_param(self.encoder)} params")
        logging.info(f"Decoder with {count_param(self.decoder)} params")
        if self.use_cls:
            logging.info(f"CLS Head with {count_param(self.cls_head)} params")

    def encode(self, x):
        input_pcl = x.double() if self.use_double else x
        B, _, N = input_pcl.shape
        device = input_pcl.device

        # normalize the point clouds: centriod and scale
        centroid = input_pcl.mean(-1) # B,3
        input_pcl = input_pcl - centroid[..., None]

        # scale initialization
        dist = torch.cdist(input_pcl.transpose(-1,-2), input_pcl.transpose(-1,-2))
        scale_0 = dist.view(B, -1).topk(5, dim=-1)[0].mean(-1)
        input_pcl = input_pcl / scale_0[:,None,None]

        # encoding
        encoder_ret = self.encoder(input_pcl)

        if len(encoder_ret) == 4:
            center_pred, pred_scale, pred_so3_feat, pred_inv_feat = encoder_ret
            centroid = center_pred.squeeze(1) + centroid
            scale = scale_0 * pred_scale
        else:
            pred_scale, pred_so3_feat, pred_inv_feat = encoder_ret
            scale = scale_0 * pred_scale
        
        embedding = {
            "z_so3": pred_so3_feat,
            "z_inv": pred_inv_feat,
            "s": scale,
            "t": centroid.unsqueeze(1),
        }

        return embedding

    def encode_fps(self, batch_pc, batch_mask, n_fps = 1):
        assert batch_pc.shape[-1] == batch_mask.shape[-1], "point cloud and mask must have same length!"
        code_list = []
        random_start = False if n_fps == 1 else True
        for pc, mask in zip(batch_pc, batch_mask):
            valid_pc = pc.T[mask.squeeze()].unsqueeze(0) # B, N, 3
            fps_pc_list = [fps(valid_pc, K=self.field_input_n, random_start_point=random_start)[0] for _ in range(n_fps)]
            fps_pc = torch.cat(fps_pc_list).transpose(-1, -2)
            embedding = self.encode(fps_pc)
            # average output embeddings
            for key in list(embedding.keys()): embedding[key] = embedding[key].mean(0, keepdim=True)
            code_list.append(embedding)
        
        batch_embedding = {}
        for key in list(code_list[0].keys()): batch_embedding[key] = torch.cat([code[key] for code in code_list], dim=0)
            
        return batch_embedding
    
    def forward(self, x):
        raise NotImplementedError()


class FieldWrapper(
    nn.Module
):  # To handle multiple decoder and wrapper for the actual decoder function
    def __init__(self, decoder, decoder_type, sdf2occ_factor=-1.0) -> None:
        super().__init__()
        self.F = decoder
        self.sdf2occ_factor = sdf2occ_factor
        self.decoder_type = decoder_type

    def forward(self, query, z_none, c, return_sdf=False):
        B, M, _ = query.shape

        z_so3, z_inv = c["z_so3"], c["z_inv"]
        scale, center = c["s"], c["t"]

        q = (query - center) / scale[:, None, None]

        inner = (q.unsqueeze(1) * z_so3.unsqueeze(2)).sum(dim=-1)  # B,C,N
        length = q.norm(dim=-1).unsqueeze(1)
        inv_query = torch.cat([inner, length], 1).transpose(2, 1)  # B,N,D

        if self.decoder_type == "inner":
            input = torch.cat([inv_query, z_inv[:, None, :].expand(-1, M, -1)], -1)
            sdf = self.F(input)
        elif self.decoder_type == "deepsdf":
            # codes = z_inv.unsqueeze(1).repeat_interleave(M, dim=1)
            input = torch.cat([z_inv.unsqueeze(1).repeat_interleave(M, dim=1), query], dim=2)
            sdf = self.F(input, 'val')
        elif self.decoder_type == "inner_deepsdf":
            input = torch.cat([z_inv[:, None, :].expand(-1, M, -1), inv_query], -1)
            sdf = self.F(input, 'val')
            # raise NotImplementedError
        elif self.decoder_type == 'inv_mlp':
            # codes = z_inv.unsqueeze(1).repeat_interleave(M, dim=1)
            input = torch.cat([z_inv.unsqueeze(1).repeat_interleave(M, dim=1), query], dim=2)
            sdf = self.F(input)
        else:
            sdf = self.F(inv_query, None, z_inv)

        if return_sdf:
            return sdf
        else:
            return dist.Bernoulli(logits=self.sdf2occ_factor * sdf)



def load_ckpt_from_log(ckpt_path):
    with open("./configs/room4cates.yaml", "r") as f:
        cfg = yaml.full_load(f)

    cfg["working_dir"] = os.getcwd()
    ckpt_list = glob.glob(osp.join(ckpt_path,'checkpoint/*latest.pt'))
    assert len(ckpt_list) == 1, " Error loading the checkpoint! "
    cfg['shape_priors']['chair']['field_pt'] = ckpt_list[0]
    
    field_cfg = glob.glob(osp.join(ckpt_path,'files_backup/*.yaml'))
    assert len(field_cfg) == 1, "config file not found of more than one config file found!"
    cfg['shape_priors']['chair']['field_cfg'] = field_cfg[0]
    
    device = torch.device("cuda")
    model = load_models_dict(cfg, device)
    
    return model

def wrap_encoder_output(outputs):
    embedding = dict()
    embedding['z_so3'] = outputs[2]
    embedding['z_inv'] = outputs[-1]
    embedding['s'] = outputs[1]
    embedding['t'] = outputs[0]
    return embedding

def mesh_from_latent(extractor, latent_code, decoder):
    
    centroid = latent_code['t']
    scale = latent_code['s']
    latent_code["t"] = torch.zeros_like(centroid)
    latent_code['s']  = torch.ones_like(scale)
    mesh = extractor.generate_from_latent(latent_code, decoder)
    # apply scale
    tsfm = np.eye(4) * scale.squeeze().item()
    tsfm[-1,-1] = 1
    # apply translation
    tsfm[:3,3] = centroid.squeeze().view(-1).detach().cpu().numpy()
    mesh.apply_transform(tsfm)
    return mesh


def slice_code_dict(code_dict, index):
    '''
    index code_dict with batch_size > 1
    '''
    return {
        'z_inv': code_dict['z_inv'][index][None],
        'z_so3': code_dict['z_so3'][index][None],
        's': code_dict['s'][index][None],
        't': code_dict['t'][index][None],
        }
    