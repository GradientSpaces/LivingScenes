"""
Code for evaluation on FlyingShapes dataset
Author: Liyuan Zhu
Date: Sep 2023
Email: liyzhu@stanford.edu
"""
import os, sys, yaml, shutil
import glob
import torch
import numpy as np
import os.path as osp
from lib_math import torch_se3
import trimesh
import point_cloud_utils as pcu
from pytorch3d.ops import sample_farthest_points as fps
from tqdm import tqdm
from lib_more.more_solver import More_Solver

from lib_more.pose_estimation import *
from pycg import vis
import logging, coloredlogs
from lib_more.utils import read_list_from_txt, load_json, load_yaml, visualize_shape_matching
from evaluate import compute_chamfer_distance, chamfer_distance_torch, compute_sdf_recall, compute_volumetric_iou


def set_logger(log_path):
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler(log_path, mode='w')
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

class FlyingShape(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        
        # load data
        # [int(shape_n.split("_")[-1]) for shape_n in sorted(os.listdir(path))]
        self.n_shape_lst = sorted(os.listdir(path))
        self.scene_lst = []
        for n_shape in self.n_shape_lst:
            n = int(n_shape.split('_')[-1])
            scene_lst = sorted(os.listdir(osp.join(path, n_shape)))
            self.scene_lst += [osp.join(path, n_shape, scene_i) for scene_i in scene_lst]


    def _load_scene(self, scene_path):
        # get scene dict
        scene_list = sorted(glob.glob(osp.join(scene_path, "*.npz")))
        scene_list = [np.load(scene_path) for scene_path in scene_list]
        return scene_list
    
    def __len__(self):
        return len(self.scene_lst)

    def __getitem__(self, idx):
        scene_path = self.scene_lst[idx]
        return self._load_scene(scene_path)


def eval_matching(dataset, solver):
    logging.info("Evaluating 3D Shape Matching on FlyingShapes")
    logging.info(f'------------------------------------')

    model = solver.model

    n_total = 0
    # method_list = ['nn', 'sinkhorn', 'sequential', 'sim3_seq', 'eq_seq']
    method = 'sequential'

    n_correct_total = 0
    n_match_total = 0
    ratio_lst = []
    for data in tqdm(dataset):
        ref_pc = torch.from_numpy(data[0]['pc']).cuda().float().transpose(-1, -2)
        rescan_lst = [torch.from_numpy(scene['pc']).cuda().float().transpose(-1, -2) for scene in data[1:]]
        with torch.no_grad():
            ref_code = model.encode(ref_pc)
            rescan_code_lst = [model.encode(rescan_pc) for rescan_pc in rescan_lst]

        for rescan_code in rescan_code_lst:
            n_obj = rescan_code['z_inv'].shape[0]
            
            matches = solver._solve_object_matching(ref_code, rescan_code, method)
            pred_match = matches['matches0']
            gt_match = torch.arange(n_obj).cuda()


            n_correct = (pred_match == gt_match).sum()
            n_match = n_obj

            n_correct_total += n_correct
            n_match_total += n_match

            ratio = n_correct / n_match
            ratio_lst.append(ratio.item())
    
    recall = n_correct_total/n_match_total * 100
    ratio_lst = np.array(ratio_lst) * 100
    scene_recall25 = (ratio_lst>=25).mean() *100
    scene_recall50 = (ratio_lst>=50).mean() *100
    scene_recall75 = (ratio_lst>=75).mean() *100
    scene_recall100 = (ratio_lst>=100).mean() *100
    logging.info(f'Object-level matching recall: {recall}')
    logging.info(f'Scene-level recall @25: {scene_recall25:.2f} | @50: {scene_recall50:.2f} | @75: {scene_recall75:.2f} | @100: {scene_recall100:.2f}')
    return


def eval_relocalization(dataset, solver):
    logging.info("Evaluating 3D Shape Registration on FlyingShapes")
    logging.info(f'------------------------------------')

    model = solver.model
    # metrics

    rre_list, rte_list, tsfm_err_list, cd_lst = [], [], [], []
    for data in tqdm(dataset):
        ref_pc = torch.from_numpy(data[0]['pc']).cuda().float().transpose(-1, -2)
        rescan_pc_lst = [torch.from_numpy(scene['pc']).cuda().float().transpose(-1, -2) for scene in data[1:2]]

        ref_tsfm = torch.from_numpy(data[0]['transform']).cuda().float()
        rescan_tsfm_lst = [torch.from_numpy(scene['transform']).cuda().float() for scene in data[1:2]]
        # with torch.no_grad():
        #     ref_code = model.encode(ref_pc)
        #     rescan_code_lst = [model.encode(rescan_pc) for rescan_pc in rescan_lst]

        for rescan_pc, rescan_tsfm in zip(rescan_pc_lst, rescan_tsfm_lst):
            gt_tsfm_lst = torch_se3.concatenate(rescan_tsfm, torch_se3.inverse(ref_tsfm))
            for ins_ref, ins_rescan, gt_tsfm in zip(ref_pc, rescan_pc, gt_tsfm_lst):
                pred_R, pred_t = solver._solve_pairwise_registration(ins_ref.T[None], ins_rescan.T[None], optim=False)
                # print()
                gt_tsfm = gt_tsfm.unsqueeze(0)
                gt_R = gt_tsfm[:, :3, :3]
                gt_t = gt_tsfm[:, :3, [3]]
                rre = rotation_error(pred_R, gt_R)
                rte = translation_error(pred_t, gt_t)

                # solve symmetry
                rre = min(rre, (180-rre).abs(), (90-rre).abs())

                # print(rre.item(), rte.item())
                pred_tsfm = torch_se3.Rt_to_SE3(pred_R, pred_t)
                # transformation error
                tsfm_err = compute_transformation_error(ins_ref.T[None], ins_rescan.T[None], pred_tsfm, gt_tsfm)

                # compute chamfer distance (downsampled by 10 because there are so many points)
                chamfer_l1 = chamfer_distance_torch(ins_ref.T[None], ins_rescan.T[None], pred_tsfm, gt_tsfm)
                
                rre_list.append(rre.item())
                rte_list.append(rte.item())
                tsfm_err_list.append(tsfm_err.item())
                cd_lst.append(chamfer_l1.item())
                
    rre_list = np.array(rre_list)
    rte_list = np.array(rte_list)
    tsfm_err_list = np.array(tsfm_err_list)
    cd_lst = np.array(cd_lst)
    
    rmse_recall = 100*(rre_list< 5).mean()
    med_rre1 = np.median(rre_list[rre_list< 5])
    med_rte1 = np.median(rte_list[rre_list< 5])
    logging.info(f"Recall [5 deg]: {rmse_recall:.2f} | RRE: {med_rre1:.2f} [deg] || RTE: {med_rte1:.2f} [m]")
    
    re_recall = 100*(rre_list<10).mean()
    med_rre2 = np.median(rre_list[rre_list<10])
    med_rte2 = np.median(rte_list[rre_list<10])
    logging.info(f"Recall [RRE<10deg]: {re_recall:.2f}| RRE: {med_rre2:.2f} [deg] || RTE: {med_rte2:.2f} [m]")
    logging.info(f"Chamfer Distance: {np.median(cd_lst):.5f}")
    logging.info(f"TE: {100* np.median(tsfm_err_list[rre_list<5]):.2f} (cm)")
    logging.info(f'------------------------------------')    
    np.savez(osp.join(solver.cfg['shape_priors']['ckpt_dir'], 'summary/flyingshapes_dict_no_optim'), rre_lst=rre_list, rte_lst=rte_list, tsfm_lst = tsfm_err_list, cd_lst = cd_lst)
    return


def eval_reconstruction(dataset, solver):
    cd_lst, iou_lst, sdf_recall_lst = [], [], []
    for data in tqdm(dataset):
        for t_scene in data[:1]:
            pc = torch.from_numpy(t_scene['pc']).cuda().float().transpose(-1, -2)
            pose = t_scene['transform']
            codes = solver.model.encode(pc)
            for i in range(pc.shape[0]):
                code = {
                    "z_inv": codes['z_inv'][i][None].detach(),
                    "z_so3": codes['z_so3'][i][None].detach(),
                    "s": codes['s'][i][None].detach(),
                    "t": codes['t'][i][None].detach(),
                }

                pred_mesh = solver._mesh_from_latent(code)
                tsfm = inverse(torch.from_numpy(pose[i])).numpy()
                tsfm_mat = np.eye(4)
                tsfm_mat[:3,:4] = tsfm
                pred_mesh.apply_transform(tsfm_mat)
                gt_mesh_path = osp.join("/home/liyuanzhu/projects/MA/data/intermediate/watertight", t_scene['class_id'][i], t_scene['obj_id'][i]+'.obj')
                gt_mesh = trimesh.load_mesh(gt_mesh_path)
                if pred_mesh.vertices.shape[0] != 0:
                    cd1, cd2 = compute_chamfer_distance(gt_mesh, pred_mesh, offset=0, scale=1)
                    sdf_recall = compute_sdf_recall(pred_mesh, gt_mesh, 0.05)
                    iou = compute_volumetric_iou(pred_mesh, gt_mesh)
                    cd_lst.append(cd1+cd2)
                    sdf_recall_lst.append(sdf_recall)
                    iou_lst.append(iou)
                    
                else:
                    iou_lst.append(0)
                    sdf_recall_lst.append(0)
    logging.info(f"Chamfer {np.mean(cd_lst):.7f}")
    logging.info(f"Mean SDF Recall: {(np.array(sdf_recall_lst)>0.7).mean()*100:.3f}")
    logging.info(f"V-iou recall: {(np.array(iou_lst)>0.5).mean()*100:.3f}")
    logging.info(f"V-iou mean: {(np.array(iou_lst)).mean()*100:.3f}")
    logging.info(f"V-iou median: {(np.median(iou_lst))*100:.3f}")
    return

if __name__ == "__main__":
    dataset_path = "/home/liyuanzhu/projects/MA/flying_shapes/dataset"
    dataset = FlyingShape(dataset_path)
    ckpt = "weights"
    solver_cfg = load_yaml("configs/more_3rscan.yaml")
    set_logger('eval_flyingshape.log')
    solver_cfg['shape_priors']['ckpt_dir'] = ckpt
    
    logging.info(f'--------Evaluation on 3RScan--------')
    solver = More_Solver(solver_cfg)
    logging.info(f'------------------------------------')

    eval_matching(dataset, solver)
    eval_relocalization(dataset, solver)
    eval_reconstruction(dataset, solver)