import os, sys
import os.path as osp
import torch, yaml
import numpy as np
from pytorch3d.ops import sample_farthest_points as fps
from pytorch3d.ops.points_alignment import iterative_closest_point, SimilarityTransform 
import torchlie as lie
import logging
from lib_math.torch_se3 import Rt_to_SE3, transform, inverse
from rich.pretty import pretty_repr
from lib_more.pose_estimation import kabsch_transformation_estimation
from lib_more.matcher_new import nn_matcher, sinkhorn_matcher, sequential_matcher, sim3_seq_matcher, eq_seq_matcher
from geomloss import SamplesLoss
import roma
from pycg import vis
import trimesh

sys.path.append(osp.abspath(osp.dirname(__file__)))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../lib_shape_prior/")))

from model_utils import load_ckpt_from_log
from lib_shape_prior.core.models.utils.occnet_utils.mesh_extractor2 import Generator3D as Generator3D_MC

class More_Solver:
    def __init__(self, cfg) -> None:
        logging.info("Configuring MoRE solver")
        logging.info("Checkpoint from "+ cfg["shape_priors"]['ckpt_dir'])
        self.cfg = cfg
        self.mesh_extractor = Generator3D_MC(**cfg["mesh_extractor"])
        # self.mesh_extractor = Generator3D_MC(**yaml.full_load(open('configs/mesh_extractor.yaml', 'r')))
        # loading model pretrained on ShapeNet
        self.model = load_ckpt_from_log(cfg["shape_priors"]['ckpt_dir'])[cfg["shape_priors"]['prior_name']]
        logging.info(pretty_repr(cfg))
        

    def _mesh_from_latent(self, latent_code):
        '''
        Generate mesh of the given latent code
        Args:
            latent code (dict{z, t, z_so3, inv})
        Return:
            mesh: trimesh
        '''
        centroid = latent_code['t'].detach().clone()
        scale = latent_code['s'].detach().clone()
        latent_code["t"] = torch.zeros_like(centroid)
        latent_code['s']  = torch.ones_like(scale)
        mesh = self.mesh_extractor.generate_from_latent(latent_code, self.model.decoder)
        # apply scale
        tsfm = np.eye(4) * scale.squeeze().item()
        tsfm[-1,-1] = 1
        # apply translation
        tsfm[:3,3] = centroid.squeeze().view(-1).detach().cpu().numpy()
        mesh.apply_transform(tsfm)
        latent_code["t"] = centroid
        latent_code["s"] = scale
        return mesh
    
    def _mesh_from_pc(self, pc):
        '''
        Args:
            pc: tensor (1, N, 3)
        Return:
            mesh: trimesh
        '''
        pc_down, _ = fps(pc, K=self.cfg['shape_priors']['n_input_point'])
        code = self.model.encode(pc_down.transpose(-1,-2))
        return self._mesh_from_latent(code)
    
    def _solve_object_matching(self, src_codes, tgt_codes, method):
        '''
        Match instances in the scene at different stages
        Args:
            src_codes: dict (embeddings of instances in source pc)
            tgt_codes: dict (embeddings of instances in target pc)
            method: optimal transport method

        Return:
            matches
        '''
        inv_codes_src = src_codes['z_inv'].detach().clone()
        inv_codes_tgt = tgt_codes['z_inv'].detach().clone()
        if method == "nn":
            return nn_matcher(inv_codes_src.T[None], inv_codes_tgt.T[None])
        elif method == "sinkhorn":
            return sinkhorn_matcher(inv_codes_src.T[None], inv_codes_tgt.T[None])
        elif method == "sequential":
            return sequential_matcher(inv_codes_src, inv_codes_tgt)
        elif method == "sim3_seq":
            return sim3_seq_matcher(src_codes, tgt_codes)
        elif method == "eq_seq":
            return eq_seq_matcher(src_codes, tgt_codes)

    def _solve_pairwise_registration(self, pc1_full, pc2_full, optim=False):
        '''
        Solve the 3d rigid transformation between pc1 and pc2, transform direction: pc1 -> pc2
        Args: 
            pc1: tensor (1, N, 3)
            pc2: tensor (1, M, 3)
            optim: use optimization (bool)
        Return:
            R: tensor (1, 3, 3)
            t: tensor (1, 3, 1)
        '''

        pc1, _ = fps(pc1_full.repeat_interleave(self.cfg['fps']['n_init'], dim=0), K=self.cfg['shape_priors']['n_input_point'])
        pc2, _ = fps(pc2_full.repeat_interleave(self.cfg['fps']['n_init'], dim=0), K=self.cfg['shape_priors']['n_input_point'])
        
        with torch.no_grad():
            code1 = self.model.encode(pc1.transpose(-1,-2))
            code2 = self.model.encode(pc2.transpose(-1,-2))

        code1_se3 = code1['z_so3'] + code1['t']
        code2_se3 = code2['z_so3'] + code2['t']
        R, t, _, _ = kabsch_transformation_estimation(code1_se3, code2_se3)

        if optim:
            with torch.no_grad():
                sdf_error1 = self.model.decoder(pc1, None, code1, return_sdf=True).abs().mean()
                sdf_error2 = self.model.decoder(pc2, None, code2, return_sdf=True).abs().mean()
            tsfm_direction = "pc1->pc2"
            
            # making initialization choice
            if sdf_error1 >= sdf_error2:
                shared_code = code2
                src_pc = pc1
                tgt_pc = pc2
                
            elif sdf_error1 < sdf_error2:
                tsfm_direction = "pc2->pc1"
                shared_code = code1
                src_pc = pc2
                tgt_pc = pc1
                R, t, res, _ = kabsch_transformation_estimation(code2_se3, code1_se3)
            
            # optimization-based registration on SE3 manifold
            g1 = lie.LieTensor(torch.cat([R,t], dim=2).detach(), lie.SE3, requires_grad=True)
            init_g = g1.new_tensor(g1)
            optim_params = [{'params': g1, 'lr': self.cfg['registration']['step_size']['so3']},
                            ]
            optimizer = torch.optim.Adam(optim_params)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,340,380], gamma=0.1)
            loss_fn_sdf = torch.nn.SmoothL1Loss()
            loss_sinkhorn = SamplesLoss(loss='sinkhorn', p=2)
            n_steps = self.cfg['registration']['n_steps']
            min_loss = 100.
            early_stop_thres = self.cfg['registration']['early_stop_threshold']

            for i in range(n_steps):
                optimizer.zero_grad()
                # iteration due to a bug in lietorch
                query_pts = g1.transform(src_pc)
                sdf_output = self.model.decoder(query_pts, None, shared_code, return_sdf=True)
                sdf_loss = loss_fn_sdf(sdf_output, torch.zeros_like(sdf_output))
                ep_loss = loss_sinkhorn(query_pts, tgt_pc) 
                loss = sdf_loss + ep_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                # print(f"Iter {i}. Loss: {loss.item(): .3f}")
                optimizer.zero_grad()

                if loss < min_loss:
                    min_loss = loss.item()
                    best_g = g1.new_tensor(g1)
                
                # stop if the change is too big
                # cur_g = g1.new_tensor(g1)
                rot_deg = roma.rotmat_geodesic_distance(g1._t[:,:3,:3], init_g._t[:,:3,:3]).mean()
                if rot_deg > early_stop_thres: break
                # del cur_g
            
            if tsfm_direction == "pc2->pc1":
                    best_g = best_g.inv()

            with lie.as_euclidean():
                R = best_g._t[...,:3]
                t = best_g._t[...,[3]]
        
            # use icp refinement
        s0 = torch.tensor([1]).float().cuda()
        icp_solution  = iterative_closest_point(pc1, pc2, init_transform=SimilarityTransform(R.transpose(-1,-2), t.squeeze(2), s0))
        R, t, _ = icp_solution[3]

        R = R.transpose(-1, -2)
        t = t.unsqueeze(2)

        return R, t
    
    def _optimize_code(self, code, pc, mask):
        valid_pc = pc.T[mask.squeeze()].squeeze()[None]
        pc, _ = fps(valid_pc, K=self.cfg['shape_priors']['n_input_point'])
        # start optimization
        optim_params = [{'params': code['z_inv'], 'lr': 1e-5},
                        {'params': code['t'], 'lr': 1e-4},
                        {'params': code['z_so3'], 'lr': 5e-4},
                        # {'params': g1, 'lr': 1e-4}
                            ]
        for param in optim_params: param['params'].requires_grad_(True).retain_grad()
        optimizer = torch.optim.Adam(optim_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160], gamma=0.1)
        loss_fn = torch.nn.MSELoss()
        n_steps = 200
        # n_steps = 300
        min_loss = 100.

        for i in range(n_steps):
            optimizer.zero_grad()
            sdf_output = self.model.decoder(pc, None, code, return_sdf=True)
            sdf_loss = loss_fn(sdf_output, torch.zeros_like(sdf_output))
            reg_loss = 0
            loss = sdf_loss + reg_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss < min_loss:
                min_loss = loss.item()
                best_code = {
                    "z_inv": code["z_inv"].detach(),
                    "z_so3": code["z_so3"].detach(),
                    "s": code["s"].detach(),
                    "t": code["t"].detach(),
                }
            # print(f"Iter {i}. Loss: {sdf_loss.item(): .3f}")
            optimizer.zero_grad()
            
        return best_code
    
    def _transform_latent(self, code, tsfm):
        '''
        Transform the equivairiant embeddings
        '''
        R = tsfm[:, :,:3]
        t = tsfm[:, :, [3]]
        new_so3 = code['z_so3'] @ R.transpose(-1, -2)
        new_t = transform(tsfm, code['t'])
        tsfm_code = {
            "z_so3": new_so3.detach().clone(),
            "z_inv": code['z_inv'].detach().clone(),
            "t": new_t.detach().clone(),
            "s": code['s'].detach().clone()
        }
        return tsfm_code
    
    def _solve_end2end(self, ref, rescan, optim=False):
        ref_pc_lst, ref_pc_full, rescan_pc_lst, rescan_pc_full = [], [], [], []
        if ref is None: return None
        for pc, mask in zip(ref['pc'],ref['pc_mask']):
            valid_pc = pc.T[mask.T.squeeze()].unsqueeze(0)
            ref_pc_full.append(valid_pc)
            fps_pc, _ = fps(valid_pc.repeat_interleave(self.cfg['fps']['n_init'], dim=0), K=self.cfg['shape_priors']['n_input_point'])
            ref_pc_lst.append(fps_pc.transpose(-1,-2))
        ref_pc_lst = torch.cat(ref_pc_lst, dim=0)

        for pc, mask in zip(rescan['pc'],rescan['pc_mask']):
            valid_pc = pc.T[mask.T.squeeze()].unsqueeze(0)
            rescan_pc_full.append(valid_pc)
            fps_pc, _ = fps(valid_pc.repeat_interleave(self.cfg['fps']['n_init'], dim=0), K=self.cfg['shape_priors']['n_input_point'])
            rescan_pc_lst.append(fps_pc.transpose(-1,-2))
        rescan_pc_lst = torch.cat(rescan_pc_lst, dim=0)

        ref_codes = self.model.encode(ref_pc_lst)
        rescan_codes = self.model.encode(rescan_pc_lst)
        end2end_dict = {
            "ref_pc_lst": ref_pc_full,
            "rescan_pc_lst": rescan_pc_full
        }
        
        matches = self._solve_object_matching(ref_codes, rescan_codes, "sequential")
        end2end_dict["matches"] = matches['matches0']
        end2end_dict["registration"] = []
        mesh_list = []
        for i, match_id in enumerate(end2end_dict["matches"]):
            if matches['matches0'][i] == -1:
                end2end_dict['registration'].append(None)
                mesh_list.append(None)
            else:
                pc1 = ref_pc_full[i]
                pc2 = rescan_pc_full[matches['matches0'][i].item()]
                R, t = self._solve_pairwise_registration(pc1, pc2, optim=optim)
                end2end_dict['registration'].append(Rt_to_SE3(R, t))

                # for i, match in enumerate(end2end_dict['matches']):
                # transform the rescan pc to ref
                if len(end2end_dict['registration'][i]) == 0: mesh_list.append(0)
                tsfm = inverse(end2end_dict['registration'][i])
                cur_code = {
                    "z_so3": rescan_codes["z_so3"][match_id][None],
                    "z_inv": rescan_codes["z_inv"][match_id][None],
                    "s": rescan_codes["s"][match_id][None],
                    "t": rescan_codes["t"][match_id][None]
                }
                new_code = self._transform_latent(cur_code, tsfm)
                new_mesh = self._mesh_from_latent(new_code)
                mesh_list.append(new_mesh)
        
        end2end_dict['mesh_lst'] = mesh_list
        return end2end_dict