from core.lib.vec_sim3.vec_layers import VecLinear
from .model_base import ModelBase
import torch
import copy
import trimesh
from torch import nn
from scipy.spatial.transform import Rotation


from core.lib.implicit_func.onet_decoder import Decoder, DecoderCBatchNorm, DecoderCat
from core.lib.implicit_func.deepsdf_decoder import DeepSDF_Decoder as Deepsdf
from core.lib.vec_sim3.vec_dgcnn import VecDGCNN, VecDGCNN_v2
from core.lib.vec_sim3.vec_dgcnn_atten import VecDGCNN_att
from core.lib.vec_sim3.dgcnn import DGCNN
from core.lib.vec_sim3.pointnet import PointNet
from core.lib.vec_sim3.pcnet import PCNet

import time
import logging
from .utils.occnet_utils import get_generator as get_mc_extractor
from .utils.ndf_utils.pcl_extractor import get_generator as get_udf_extractor
from .utils.misc import cfg_with_default, count_param
from torch import distributions as dist
import numpy as np

from core.models.utils.oflow_eval.evaluator import MeshEvaluator
from core.models.utils.oflow_common import eval_iou


class Model(ModelBase):
    def __init__(self, cfg):
        network = SIM3Recon(cfg)
        super().__init__(cfg, network)

        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_recon_uni",
                "loss_recon_nss",
                "loss_kl",
                "metric_recon_uni_error",
                "metric_recon_nss_error",
                "metric_center_error",
                "iou",
                "acc",
                "acc_nss",
                "acc_far",
                "loss_cls",
                "metric_bs_cls_acc",
            ]
            + ["loss_s", "loss_t", "metric_t"],
            "image": ["rendered_fig_list"],
            "mesh": ["mesh", "input"],
            "hist": [
                "loss_recon_i",
                "iou_i",
                "scale",
                "metric_t_i",
                "acc_i",
                "acc_nss_i",
                "acc_far_i",
                "loss_kl_i",
            ],
            "xls": ["running_metric_report", "results"],
        }

        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        self.iou_threshold = cfg["evaluation"]["iou_threshold"]
        self.evaluator = MeshEvaluator(100000)

        self.nss_th = cfg_with_default(cfg, ["model", "loss_th"], 1.0)
        self.df_acc_th = cfg_with_default(cfg, ["evaluation", "df_acc_threshold"], 0.01)

        self.use_udf = cfg_with_default(cfg, ["model", "use_udf"], False)
        if self.use_udf:
            self.pcl_extractor = get_udf_extractor(cfg)
        else:
            self.mesh_extractor = get_mc_extractor(cfg)

    def generate_mesh(self, embedding):
        assert not self.use_udf
        net = self.network.module if self.__dataparallel_flag__ else self.network

        mesh = self.mesh_extractor.generate_from_latent(c=embedding, F=net.decode)
        if mesh.vertices.shape[0] == 0:
            mesh = trimesh.primitives.Box(extents=(1.0, 1.0, 1.0))
            logging.warning("Mesh extraction fail, replace by a place holder")
        return mesh

    def generate_dense_surface_pts(self, embedding):
        net = self.network.module if self.__dataparallel_flag__ else self.network

        for param in self.network.network_dict["decoder"].parameters():
            param.requires_grad = False

        pcl = self.pcl_extractor.generate_from_latent(c=embedding, F=net.decode)
        if pcl is None:
            logging.warning("Dense PCL extraction fail, replace by a point at origin")
            pcl = np.array([[0.0, 0.0, 0.0]])

        for param in self.network.network_dict["decoder"].parameters():
            param.requires_grad = True

        return pcl

    def _postprocess_after_optim(self, batch):
        if "occ_hat_iou" in batch.keys() and not self.use_udf:
            # IOU is only directly computable when using sdf
            report = {}
            occ_pred = batch["occ_hat_iou"].unsqueeze(1).detach().cpu().numpy()
            occ_gt = batch["model_input"]["eval.points.occ"].unsqueeze(1).detach().cpu().numpy()
            iou = eval_iou(occ_gt, occ_pred, threshold=self.iou_threshold)  # B,T_all
            # make metric tensorboard
            batch["iou"] = iou.mean()
            batch["iou_i"] = torch.from_numpy(iou).reshape(-1)
            # make report
            report["iou"] = iou.mean(axis=1).tolist()
            batch["running_metric_report"] = report
        if "df_hat" in batch.keys():
            df_gt = batch["model_input"]["eval.points.value"].unsqueeze(1).detach().cpu().numpy()
            df_gt = abs(df_gt.squeeze(1))
            df_hat = abs(batch["df_hat"]).detach().cpu().numpy()
            df_error = abs(df_gt - df_hat)
            df_correct = df_error < self.df_acc_th
            nss_mask = (df_gt < self.nss_th).astype(np.float)
            far_mask = 1.0 - nss_mask
            df_acc = df_correct.sum(-1) / df_correct.shape[1]
            df_acc_nss = (df_correct * nss_mask).sum(-1) / (nss_mask.sum(-1) + 1e-6)
            df_acc_far = (df_correct * far_mask).sum(-1) / (far_mask.sum(-1) + 1e-6)
            batch["acc_i"], batch["acc"] = df_acc, df_acc.mean()
            batch["acc_nss_i"], batch["acc_nss"] = df_acc_nss, df_acc_nss.mean()
            batch["acc_far_i"], batch["acc_far"] = df_acc_far, df_acc_far.mean()

        if "z_so3" in batch.keys():
            self.network.eval()
            phase = batch["model_input"]["phase"]
            n_batch = batch["z_so3"].shape[0]
            # TEST_RESULT = {}
            with torch.no_grad():
                batch["mesh"] = []
                rendered_fig_list = []
                for bid in range(n_batch):
                    start_t = time.time()
                    embedding = {
                        "z_so3": batch["z_so3"][bid : bid + 1],
                        "z_inv": batch["z_inv"][bid : bid + 1],
                        "s": batch["s"][bid : bid + 1],
                        "t": batch["t"][bid : bid + 1],
                    }

                    if self.use_udf:  # generate dense pcl
                        from .utils.viz_udf_render import viz_input_and_recon

                        dense_pcl = self.generate_dense_surface_pts(embedding=embedding)
                        batch["dense_pcl"] = dense_pcl
                        rendered_fig = viz_input_and_recon(
                            input=batch["input"][bid].detach().cpu().numpy(), output=dense_pcl
                        )
                        # imageio.imsave("./debug/dbg.png", rendered_fig)
                        rendered_fig_list.append(rendered_fig.transpose(2, 0, 1)[None, ...])
                        # print()
                        # todo: render this pcl to viz, tensorboard is bad
                    else:  # generate mesh
                        mesh = self.generate_mesh(embedding=embedding)
                        batch["mesh"].append(mesh)
                    if self.viz_one and not phase.startswith("test"):
                        break
                if len(rendered_fig_list) > 0:
                    batch["rendered_fig_list"] = torch.Tensor(
                        np.concatenate(rendered_fig_list, axis=0)
                    )  # B,3,H,W
        return batch


class SIM3Recon(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        self.encoder_64_flag = cfg_with_default(cfg, ["model", "encoder_64"], True)
        # assert self.encoder_64_flag, "should use 64"

        self.decoder_type = cfg_with_default(cfg, ["model", "decoder_type"], "decoder")
        decoder_class = {"decoder": Decoder, 
                         "cbatchnorm": DecoderCBatchNorm, 
                         "inner": DecoderCat, 
                         "deepsdf": Deepsdf,
                         "inner_deepsdf": Deepsdf,
                         "inv_mlp": DecoderCat
                         }[self.decoder_type]

        self.encoder_type = cfg_with_default(cfg, ["model", "encoder_type"], "sim3pointres")
        encoder_class = {
            "vecdgcnn": VecDGCNN,
            "vecdgcnn2": VecDGCNN_v2,
            "vecdgcnn_atten": VecDGCNN_att,
            "dgcnn": DGCNN,
            "pointnet": PointNet,
            "pcnet": PCNet
        }

        encoder = encoder_class[self.encoder_type](**cfg["model"]["encoder"])
        if self.encoder_64_flag:
            encoder = encoder.double()
        self.network_dict = torch.nn.ModuleDict(
            {
                "encoder": encoder,
                "decoder": decoder_class(**cfg["model"]["decoder"]),
            }
        )

        self.decoder_use_pe = cfg_with_default(cfg, ["model", "use_pe"], False)
        if self.decoder_use_pe:
            self.pe_src = cfg["model"]["pe_src"]
            self.pe_pow = cfg["model"]["pe_pow"]
            self.pe_sigma = np.pi * torch.pow(2, torch.linspace(0, self.pe_pow - 1, self.pe_pow))
            self.network_dict["pe_projector"] = VecLinear(
                cfg["model"]["encoder"]["c_dim"], self.pe_src
            )

        self.use_cls = cfg_with_default(cfg, ["model", "use_cls"], False)
        if self.use_cls:
            self.network_dict["cls_head"] = nn.Sequential(
                nn.Linear(cfg["model"]["encoder"]["c_dim"], cfg["model"]["encoder"]["c_dim"]),
                nn.Sigmoid(),
                nn.Linear(cfg["model"]["encoder"]["c_dim"], cfg["model"]["encoder"]["c_dim"]),
                nn.Sigmoid(),
                nn.Linear(cfg["model"]["encoder"]["c_dim"], cfg["model"]["num_cates"]),
            )
            self.w_cls = cfg_with_default(cfg["model"], ["w_cls"], 1.0)
            self.criterion_cls = nn.CrossEntropyLoss()

        self.w_s = cfg_with_default(cfg["model"], ["w_s"], 0.0)
        self.w_t = cfg_with_default(cfg["model"], ["w_t"], 0.0)
        self.w_recon = cfg_with_default(cfg["model"], ["w_recon"], 1.0)

        self.sdf2occ_factor = cfg_with_default(cfg, ["model", "sdf2occ_factor"], -1.0)
        self.w_uni = cfg_with_default(cfg, ["model", "w_uni"], 1.0)
        self.w_nss = cfg_with_default(cfg, ["model", "w_nss"], 1.0)

        self.loss_th = cfg_with_default(cfg, ["model", "loss_th"], 1.0)
        self.loss_near_lambda = cfg_with_default(cfg, ["model", "loss_near_lambda"], 1.0)
        self.loss_far_lambda = cfg_with_default(cfg, ["model", "loss_far_lambda"], 0.1)

        self.training_centroid_aug_std = cfg["model"]["center_aug_std"]

        self.use_udf = cfg_with_default(cfg, ["model", "use_udf"], False)
        if self.use_udf:
            logging.info("Use UDF instead of SDF")

        self.rot_aug = cfg_with_default(cfg, ["model", "rot_aug"], False)
        if self.rot_aug:
            logging.warning(f"Use Rot Aug, this should only happen for ablation study!")

        count_param(self.network_dict)

        return

    def forward(self, input_pack, viz_flag):
        output = {}
        phase, epoch = input_pack["phase"], input_pack["epoch"]

        # prepare inputs
        input_pcl = input_pack["inputs"].transpose(2, 1)
        query = torch.cat([input_pack["points.uni"], input_pack["points.nss"]], dim=1)
        B, _, N = input_pcl.shape
        device = input_pcl.device

        if self.rot_aug and (phase == "train" or phase=="val"):
            # ! only aug during training
            random_R = [torch.from_numpy(Rotation.random().as_matrix()) for _ in range(B)]
            random_R = torch.stack(random_R, 0).to(device).type(input_pcl.dtype)
            input_pcl = torch.einsum("bij,bjn->bin", random_R, input_pcl)
            if self.decoder_type not in ['deepsdf', 'inv_mlp']: # deepsdf decode shapes in canonical space
                query = torch.einsum("bij,bnj->bni", random_R, query)

        # augmentation on the center
        centroid = input_pcl.mean(-1)  # B,3
        if phase == "train":
            noise = torch.normal(
                mean=0.0, std=self.training_centroid_aug_std * torch.ones_like(centroid)
            )
            centroid = centroid + noise
        input_pcl = input_pcl - centroid[..., None]

        # encoding
        if self.encoder_64_flag:
            input_pcl = input_pcl.double()
        encoder_ret = self.network_dict["encoder"](input_pcl)
        if len(encoder_ret) == 4:
            center_pred, pred_scale, pred_so3_feat, pred_inv_feat = encoder_ret
            centroid = center_pred.squeeze(1) + centroid
        else:
            pred_scale, pred_so3_feat, pred_inv_feat = encoder_ret
        if self.encoder_64_flag:
            pred_scale = pred_scale.float()
            pred_so3_feat, pred_inv_feat = pred_so3_feat.float(), pred_inv_feat.float()

        loss_scale = abs(pred_scale - 1.0).mean()
        loss_center = centroid.norm(1, dim=-1).mean()
        error_center_i = centroid.norm(dim=-1)

        embedding = {
            "z_so3": pred_so3_feat,
            "z_inv": pred_inv_feat,
            "s": pred_scale,
            "t": centroid.unsqueeze(1),
        }

        if phase.startswith("test") or viz_flag:
            output["z_so3"] = pred_so3_feat
            output["z_inv"] = pred_inv_feat
            output["s"], output["t"] = (pred_scale.detach(), centroid.unsqueeze(1).detach())
            output["input"] = input_pack["inputs"]
        if phase.startswith("test"):
            return output

        N_uni = input_pack["points.uni"].shape[1]
        sdf_hat = self.decode(  # SDF must have nss sampling
            query=query,
            phase=phase,
            c=embedding,
            return_sdf=True
        )
        sdf_gt = torch.cat([input_pack["points.uni.value"], input_pack["points.nss.value"]], dim=1)
        if self.use_udf:
            sdf_gt, sdf_hat = abs(sdf_gt), abs(sdf_hat)

        sdf_error_i = abs(sdf_hat - sdf_gt)
        sdf_near_mask = (sdf_error_i < self.loss_th).float().detach()
        sdf_loss_i = (
            sdf_error_i * sdf_near_mask * self.loss_near_lambda
            + sdf_error_i * (1.0 - sdf_near_mask) * self.loss_far_lambda
        )
        uni_sdf_loss_i, nss_sdf_loss_i = sdf_loss_i[:, :N_uni], sdf_loss_i[:, N_uni:]
        uni_sdf_error_i, nss_sdf_error_i = sdf_error_i[:, :N_uni], sdf_error_i[:, N_uni:]
        uni_sdf_loss, nss_sdf_loss = uni_sdf_loss_i.mean(), nss_sdf_loss_i.mean()
        uni_sdf_error, nss_sdf_error = uni_sdf_error_i.mean(), nss_sdf_error_i.mean()

        if self.use_cls:
            cls_logits = self.network_dict["cls_head"](pred_inv_feat)
            cls_pred = torch.softmax(cls_logits, -1)
            cls_gt = input_pack["class"].long()
            loss_cls = self.criterion_cls(cls_pred, cls_gt)
            # ! warning, this is not the metric, need to stat outside to metric
            cls_correct = cls_pred.argmax(-1) == cls_gt
            cls_bs_acc = cls_correct.float().mean()

        output["batch_loss"] = (
            uni_sdf_loss * self.w_uni + nss_sdf_loss * self.w_nss + self.w_s * loss_scale
        )
        if self.w_t > 0.0:
            output["batch_loss"] = output["batch_loss"] + self.w_t * loss_center
            output["loss_t"] = loss_center.detach()
            output["metric_t"] = error_center_i.mean().detach()
            output["metric_t_i"] = error_center_i.detach()
        if self.use_cls:
            output["batch_loss"] = output["batch_loss"] + self.w_cls * loss_cls
            output["loss_cls"] = loss_cls.detach()
            output["metric_bs_cls_acc"] = cls_bs_acc.detach()

        output["loss_recon_uni"] = uni_sdf_loss.detach()
        output["loss_recon_nss"] = nss_sdf_loss.detach()
        output["metric_recon_uni_error"] = uni_sdf_error.detach()
        output["metric_recon_nss_error"] = nss_sdf_error.detach()
        output["metric_recon_uni_error_i"] = uni_sdf_error_i.detach().reshape(-1)
        output["metric_recon_nss_error_i"] = nss_sdf_error_i.detach().reshape(-1)

        output["loss_s"] = loss_scale.detach()
        output["scale"] = pred_scale.reshape(-1).detach()

        if phase.startswith("val"):  # add eval
            if self.use_udf:
                output["df_hat"] = self.decode(
                    input_pack["eval.points"].reshape(B, -1, 3), None, embedding, True
                )
            else:
                output["occ_hat_iou"] = self.decode(
                    input_pack["eval.points"].reshape(B, -1, 3), None, embedding
                ).probs

        return output

    def positional_encoder(self, x):
        device = x.device
        y = torch.cat(
            [
                x[..., None],
                torch.sin(x[:, :, :, None] * self.pe_sigma[None, None, None].to(device)),
                torch.cos(x[:, :, :, None] * self.pe_sigma[None, None, None].to(device)),
            ],
            dim=-1,
        )
        return y

    def decode(self, query, phase, c, return_sdf=False):
        B, M, _ = query.shape

        z_so3, z_inv = c["z_so3"], c["z_inv"]
        scale, center = c["s"], c["t"]

        q = (query - center) / scale[:, None, None]
        inner = (q.unsqueeze(1) * z_so3.unsqueeze(2)).sum(dim=-1)  # B,C,N
        length = q.norm(dim=-1).unsqueeze(1)
        inv_query = torch.cat([inner, length], 1).transpose(2, 1)  # B,N,D
    
        if self.decoder_use_pe:
            coordinate = self.network_dict["pe_projector"](z_so3)  # B,PE_C,3
            pe_inner = (q.unsqueeze(1) * coordinate.unsqueeze(2)).sum(dim=-1)  # B,PE_C,N
            pe_query = self.positional_encoder(pe_inner)
            pe_query = pe_query.transpose(-2, -1).reshape(B, -1, M)
            inv_query = torch.cat([inv_query, pe_query.transpose(2, 1)], 2)
            
        if self.decoder_type == "inner":
            input = torch.cat([inv_query, z_inv[:, None, :].expand(-1, M, -1)], -1)
            sdf = self.network_dict["decoder"](input)
        elif self.decoder_type == "deepsdf":
            # codes = z_inv.unsqueeze(1).repeat_interleave(M, dim=1)
            input = torch.cat([z_inv.unsqueeze(1).repeat_interleave(M, dim=1), query], dim=2)
            sdf = self.network_dict["decoder"](input, phase)
        elif self.decoder_type == "inner_deepsdf":
            input = torch.cat([z_inv[:, None, :].expand(-1, M, -1), inv_query], -1)
            sdf = self.network_dict["decoder"](input, phase)
            # raise NotImplementedError
        elif self.decoder_type == 'inv_mlp':
            # codes = z_inv.unsqueeze(1).repeat_interleave(M, dim=1)
            input = torch.cat([z_inv.unsqueeze(1).repeat_interleave(M, dim=1), query], dim=2)
            sdf = self.network_dict["decoder"](input)
        else:
            sdf = self.network_dict["decoder"](inv_query, None, z_inv)

        if return_sdf:
            return sdf
        else:
            return dist.Bernoulli(logits=self.sdf2occ_factor * sdf)
