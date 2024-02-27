"""
Code for evaluation on 3RScan dataset
Author: Liyuan Zhu
Date: Sep 2023
Email: liyzhu@stanford.edu
"""
import os, sys, yaml, shutil
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
from evaluate import compute_chamfer_distance, chamfer_distance_torch, compute_sdf_recall

# category mapping between ShapeNet and RIO
SHAPENET_CATE = ["chair", "table", "bench", "sofa", "pillow", "bed", "trash_bin"]
RIO_CATE = [
    ["dinning chair", "rocking chair", "armchair", "chair"], # chair
    ["couching table", "dining table", "computer desk", "round table", "side table", "stand", "desk", "coffee table"], # table
    ["bench"], # bench
    ["sofa", "sofa chair", "couch", "ottoman", "footstool"], # sofa
    ["cushion", "pillow"], # pillow
    ["bed"], # bed
    ["trash can"], # trash bin
]

def get_shapenet_category(rio_cate):
    for shapenet_cate, rio_list in zip(SHAPENET_CATE, RIO_CATE):
        if rio_cate in rio_list: return shapenet_cate
    return "others"

def set_logger(log_path):
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler(log_path, mode='w')
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

# define the dataset
class Dataset_3RScan(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.root_path = cfg["root_path"]
        self.split = cfg["split"]
        self.data_path = os.path.join(self.root_path, f'{self.split}_set')
        self.category_list = read_list_from_txt(cfg["category_list"])
        self.n_point_per_instance = cfg["n_point_per_instance"]
        
        # read the data list
        self.scan_list = os.listdir(self.data_path)
        
        # read the split indices
        f = open(os.path.join(self.root_path, '..', f'splits/{self.split}.txt'), "r")
        self.split_indices = f.read().splitlines()
        
        # read scene json
        scene_json = load_json(os.path.join(self.root_path, '3RScan.json'))
        self.scene_list = [ scene for scene in scene_json if scene['reference'] in self.split_indices ]
        self.use_gt_mask = cfg["use_gt_mask"]
        if not self.use_gt_mask:
            self.mask_name = cfg["mask_name"]
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        raise NotImplementedError
    
    def _heterogeneous_batching(self, pc_list):
        '''
        For batching point clouds with different number of points
        '''
        
        max_n_points = max([pc.shape[-1] for pc in pc_list])
        pc_batch_list = []
        mask_batch_list = []
        for pc in pc_list:
            B,_,n_points = pc.shape
            void_points = torch.zeros(B, 3, max_n_points - n_points).to(pc.device)
            padded_pc = torch.cat((pc, void_points), dim=2)
            mask = torch.cat((torch.ones(B, 1, n_points), torch.zeros(B, 1, max_n_points - n_points)), dim=2).bool()
            pc_batch_list.append(padded_pc)
            mask_batch_list.append(mask)
        batch_pc = torch.cat(pc_batch_list, dim=0).to(pc.device)
        batch_mask = torch.cat(mask_batch_list, dim=0).to(pc.device)
        return batch_pc, batch_mask
    
    def _load_scan(self, scan_id):
        scan_path = osp.join(self.data_path, scan_id)
        semseg_list = load_json(osp.join(scan_path, 'semseg.v2.json'))['segGroups']
        scan_pc = pcu.load_mesh_v(osp.join(scan_path, 'pointcloud.instances.align.ply'))
        if self.use_gt_mask:
            pc_labels = np.load(osp.join(scan_path, 'pointcloud.labels.npz'), allow_pickle=True)
        else:
            pc_labels = np.load(osp.join(scan_path, self.mask_name), allow_pickle=True)
        pc_labels_gt = np.load(osp.join(scan_path, 'pointcloud.labels.npz'), allow_pickle=True)
        # filter out the instances that are not in the category list
        pc_list = []
        id_list = []
        center_list = []
        bg_pc = []
        label_list = []
        full_gt_id_list = []
        z_max = -100 # trim the top for visualization
        for instance in semseg_list:
            
            if instance['label'] in self.category_list:
                
                shapenet_label = get_shapenet_category(instance['label'])
                label_list.append((instance['objectId'], instance['label'], shapenet_label))
                # extract the instance pc from scene
                instance_id = torch.from_numpy(np.array(int(instance['objectId']))).cuda().reshape(-1)
                instance_pc = scan_pc[pc_labels['objectId'] == instance['objectId']]
                full_gt_id_list.append(instance_id)
                if len(instance_pc) == 0: continue
                if instance_pc[...,-1].max()>z_max: z_max = instance_pc[:,-1].max()
                if instance_pc.shape[0] < 1024: continue
                # to tensor
                instance_pc = torch.from_numpy(instance_pc).float().cuda().unsqueeze(0)
                instance_center = instance_pc.mean(dim=1) # compute the center of gravity for each instance, normalize
                
                # instance_pc, _ = fps(instance_pc, K=self.n_point_per_instance)
                pc_list.append(instance_pc.permute(0, 2, 1))
                id_list.append(instance_id)
                center_list.append(instance_center)
        for instance in semseg_list:
            if instance['label'] not in self.category_list:
                instance_pc = scan_pc[pc_labels['objectId'] == instance['objectId']]
                instance_pc = instance_pc[instance_pc[:,2]<z_max]
                bg_pc.append(instance_pc)
        
        if len(pc_list) == 0: return None

        batch_pc, batch_pc_mask = self._heterogeneous_batching(pc_list)
        bg_pc = np.concatenate(bg_pc,axis=0)[::5]

        instance = {
            "pc": batch_pc,
            "pc_mask": batch_pc_mask,
            "objectId": torch.cat(id_list, dim=0),
            "bg_pc": bg_pc,
            "id_label": label_list, # (object_id, rio_label, shapenet_label)
            "full_objectId": torch.cat(full_gt_id_list)
        }

        return instance
    
    def _get_instance_tsfm(self, instance_id, scan_id):
        return
    
    def _get_scene(self, idx): 
        # get the instance pc, label, association and transformation of the scene
        assert idx < len(self.scene_list), "scene index out of range!"
        scene_dict = self.scene_list[idx]
        reference = self._load_scan(scene_dict["reference"])
        # rescan_list = [self._load_scan(scan["reference"]) for scan in scene_dict['scans']]
        rescan_list = []
        # rule out none scans
        for scan in scene_dict['scans']:
            rescan = self._load_scan(scan["reference"])
            
            if rescan is not None:
                scene_tsfm = torch.Tensor(scan['transform']).cuda().reshape(1, 4, 4).transpose(-1, -2)
                moving_id_lst = []
                static_id_lst = []
                for rigid in scan['rigid']: 
                    obj_tsfm = inverse(torch.Tensor(rigid['transform']).cuda().reshape(1, 4, 4).transpose(-1, -2)) # obj from rescan to ref
                    rot_diff = rotation_error(obj_tsfm[:,:3,:3], scene_tsfm[:,:3,:3])
                    t_diff = translation_error(obj_tsfm[:,:3,3], scene_tsfm[:,:3,3])
                    if rot_diff>1 or t_diff > 0.05:
                        moving_id_lst.append(rigid['instance_reference'])
                    else: static_id_lst.append(rigid['instance_reference'])
                rescan['moving_ids'] = torch.Tensor(moving_id_lst).cuda()
                rescan['static_ids'] = torch.Tensor(static_id_lst).cuda()
                rescan['rescan2ref_tsfm'] = scene_tsfm
                rescan_list.append(rescan)
        
        return reference, rescan_list

def disambiguiate(pred, gt, ambiguity):
    def find_next(chain_list, pair_list):
        for pair in pair_list: # find next
                if chain_list[-1] == pair[0]:
                    return pair
    def find_all(id, ambiguity):
        # loop over all possible pairs recursively to find all pairs
        pair_list = []
        for pairs in ambiguity:
            pair_list += [(pair['instance_source'], pair['instance_target'], pair['transform']) for pair in pairs]

        chain_list = []
        tsfm_list = []
        
        for pair in pair_list: 
            if pair[0] == id: # initialize id
                chain_list.append(pair[1])
                tsfm_list.append(np.array(pair[2]).reshape(4,4).T)
        
        max_iter = 200
        c_iter = 0
        if len(chain_list) != 0:
            while c_iter < max_iter:
                new_pair = find_next(chain_list, pair_list)
                if new_pair[1] == id: break
                else:
                    chain_list.append(new_pair[1]) # append next
                    tsfm_list.append(np.array(new_pair[2]).reshape(4,4).T @ tsfm_list[-1]) # append chained transform
                c_iter += 1

        return chain_list, tsfm_list
    
    pair_list = []
    for pairs in ambiguity:
        pair_list += [(pair['instance_source'], pair['instance_target']) for pair in pairs]
    
    for i in range(gt.shape[0]):
        # pair = (pred[i].item(), gt[i].item())
        chain_list, tsfm_list = find_all(pred[i].item(), ambiguity)
        if gt[i].item() in chain_list:
            pred[i] = gt[i]
    return pred

def eval_3rscan_matching(cfg_path, solver, visualize=False):
    logging.info("Evaluating 3D Shape Matching on 3RSCan")
    logging.info(f'------------------------------------')
    with open(cfg_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    dataset_3rscan = Dataset_3RScan(data_cfg)
    model = solver.model
    
    # evaluate matching
    n_total = 0
    method_list = ['sequential']
    n_correct_list = np.zeros(len(method_list))

    scene_level_total = np.zeros(3) # recall @.25, @.5, @.75
    scene_level_count = np.zeros(3)

    n_total_dynamic, n_correct_dynamic = 0, 0
    n_total_static, n_correct_static = 0, 0
    
    
    # loop over the dataset
    for i_s, scene in tqdm(enumerate(dataset_3rscan.scene_list)):
        ref, rescan_list = dataset_3rscan._get_scene(i_s)
        
        if len(rescan_list) == 0 or ref is None: continue

        with torch.no_grad():
            
            ref_codes = model.encode_fps(ref['pc'], ref['pc_mask'])
            rescan_code_lists = [model.encode_fps(rescan["pc"], rescan['pc_mask']) for rescan in rescan_list]
            for rescan, rescan_codes, sg in zip(rescan_list, rescan_code_lists, scene['scans']):
                # get indices of moving and static object
                moving_id_lst = []
                static_id_lst = []
                scene_tsfm = torch.Tensor(sg['transform']).cuda().reshape(1, 4, 4).transpose(-1, -2)
                for rigid in sg['rigid']: 
                    obj_tsfm = inverse(torch.Tensor(rigid['transform']).cuda().reshape(1, 4, 4).transpose(-1, -2)) # obj from rescan to ref
                    rot_diff = rotation_error(obj_tsfm[:,:3,:3], scene_tsfm[:,:3,:3])
                    t_diff = translation_error(obj_tsfm[:,:3,3], scene_tsfm[:,:3,3])
                    if rot_diff>1 or t_diff > 0.05:
                        moving_id_lst.append(rigid['instance_reference'])
                    else: static_id_lst.append(rigid['instance_reference'])

                for i, method in enumerate(method_list):
                    match_score = solver._solve_object_matching(ref_codes, rescan_codes, method)

                    matched_indices_0 = rescan['objectId'][match_score['matches0']]
                
                    valid_mask = torch.tensor([id in rescan['objectId'] for id in ref['objectId']]).cuda()
                    full_mask = torch.tensor([id in rescan['full_objectId'] for id in ref['full_objectId']]).cuda()
                    # mask of preditions for valid pairs
                    pred_mask = ((match_score['matches0']) != -1)

                    # disambiguiate
                    if len(scene['ambiguity'])!=0:
                        matched_indices_0 = disambiguiate(matched_indices_0.view(-1), ref['objectId'], scene['ambiguity'])
                    # print(pred_mask.shape, matched_indices_0.shape)
                    matched_indices_0[~pred_mask] = -1

                    if visualize and i_s>=0: visualize_shape_matching(ref, rescan, matched_indices_0, rescan["objectId"]) 

                    # compute metrics
                    if len(matched_indices_0.size()) != 0:
                        
                        n_match = valid_mask.sum().item()
                        full_match = full_mask.sum().item()
                        n_correct = (matched_indices_0 == ref['objectId'])[valid_mask].sum()
                        n_correct_list[i] = n_correct_list[i] + n_correct
                        # n_total = n_total + n_match
                        n_total += n_match

                        # scene-level recall
                        scene_level_total += 1
                        ratio = n_correct/n_match
                        # print(ratio.item())
                        if  ratio >=0.75: scene_level_count[:] +=1
                        elif ratio >=0.5: scene_level_count[1:] +=1
                        elif ratio >=0.25: scene_level_count[2:] +=1

                        # dynamic, static
                        moving_mask = torch.Tensor([True if objid in moving_id_lst else False for objid in ref['objectId']]).cuda().bool()
                        static_mask = ~moving_mask
                        n_total_dynamic += (valid_mask & moving_mask).sum()
                        n_total_static += (valid_mask & static_mask).sum()
                        # print((valid_mask & moving_mask).sum().item(), (valid_mask & static_mask).sum().item(), n_match)
                        n_correct_dynamic += (matched_indices_0 == ref['objectId'])[valid_mask & moving_mask].sum()
                        n_correct_static += (matched_indices_0 == ref['objectId'])[valid_mask & static_mask].sum()
                        
                            
    n_total = n_total/len(n_correct_list)
    precision = n_correct_list/n_total * 100

    scene_recall = scene_level_count / scene_level_total * 100

    dynamic_recall = n_correct_dynamic / n_total_dynamic * 100
    static_recall = n_correct_static / n_total_static * 100

    logging.info(f'Object-level matching recall:')
    for method, pre in zip(method_list, precision): logging.info(f"{method} : (all) {pre:.2f} | (static) {static_recall.item():.2f} | (dynamic) {dynamic_recall.item():.2f}") 
    
    logging.info(f'Scene-level Hits Recall: @75 {scene_recall[0]:.2f} | K@50 {scene_recall[1]:.2f} | K@25 {scene_recall[2]:.2f}')

    # logging.info(f" NN Precision: {100*precision[0].item():.2f} | Sinkhorn Precision: {100*precision[1].item():.2f} | Sequential Precision: {100*precision[2].item():.2f}")
    logging.info(f'------------------------------------')

def evaluate_3rscan_relocalization(cfg_path, solver, visualize=False):
    '''
    evaluate relocalization/pose estimation/registration
    '''
    logging.info(f'Evaluating Instance Relocalization')
    logging.info(f'------------------------------------')
    
    data_cfg = load_yaml(cfg_path)
    dataset_3rscan = Dataset_3RScan(data_cfg)

    # metrics
    rre_list, rte_list, tsfm_err_list, shape_lst, cd_lst = [], [], [], [], []
    for i, scene in tqdm(enumerate(dataset_3rscan.scene_list)):
        ref, rescan_list = dataset_3rscan._get_scene(i)
        if ref is None: continue
        for rescan, sg in zip(rescan_list, scene['scans']):
            # transform the rescan to its original coordinates
            scene_tsfm = torch.Tensor(sg['transform']).cuda().reshape(1, 4, 4).transpose(-1, -2) # from rescan to reference
            # B, _, N = rescan['pc'].shape
            # if visualize:
            #     rescan_pc = inverse_3d_transform(scene_tsfm) @ torch.cat([rescan['pc'], torch.ones([B,1,N]).to(rescan['pc'].device)], dim=1)
            # if not visualize:
            rescan['pc'] = transform(inverse(scene_tsfm), rescan['pc'].transpose(-1, -2)).transpose(-1, -2)

            
            if visualize:
                ref_ins_list, rescan_ins_list, pred_pc_list, gt_pc_list = [], [], [], []
                # bg_list = [ref['bg_pc'], rescan['bg_pc']]
            for rigid_tsfm in sg['rigid']:
                
                # sanity check 
                if rigid_tsfm['instance_reference'] in ref['objectId'] and rigid_tsfm['instance_rescan'] in rescan['objectId']:
                    # instance in ref CS -> ins in rescan CS
                    gt_tsfm = torch.Tensor(rigid_tsfm['transform']).reshape(-1, 4, 4).transpose(-1,-2).cuda() # transform: instance from refenrence to rescan
                    # gt_tsfm = scene_tsfm @ ref_to_rescan_tsfm
                    gt_R = gt_tsfm[:, :3, :3]
                    gt_t = gt_tsfm[:, :3, [3]]
                    # get instance pc of current id
                    ref_ids = torch.where(ref['objectId'] == rigid_tsfm['instance_reference'])
                    rescan_ids = torch.where(rescan['objectId']== rigid_tsfm['instance_rescan'])
                    # get instance pc
                    instance_ref = ref['pc'][ref_ids].transpose(-1, -2)[ref['pc_mask'][ref_ids].transpose(-1,-2).squeeze(2)].unsqueeze(0)
                    instance_rescan = rescan['pc'][rescan_ids].transpose(-1, -2)[rescan['pc_mask'][rescan_ids].transpose(-1,-2).squeeze(2)].unsqueeze(0)
                    
                    pred_R, pred_t = solver._solve_pairwise_registration(instance_ref, instance_rescan, optim=True)

                    # relative rotation and translation error
                    rre = rotation_error(pred_R, gt_R)
                    rte = translation_error(pred_t, gt_t)

                    # resolve symmetry
                    if rigid_tsfm['symmetry'] == 1:
                        m = 2
                        rre = min([rre, abs(180 - rre)])
                    elif rigid_tsfm['symmetry'] == 2:
                        m = 4
                        rre = min(rre, abs(180 - rre), abs(90-rre))
                    
                    # print(rre.item(), rte.item())
                    pred_tsfm = torch_se3.Rt_to_SE3(pred_R, pred_t)
                    # transformation error
                    tsfm_err = compute_transformation_error(instance_ref, instance_rescan, pred_tsfm, gt_tsfm)

                    # compute chamfer distance (downsampled by 10 because there are so many points)
                    chamfer_l1 = chamfer_distance_torch(instance_ref[:,::10], instance_rescan[:,::10], pred_tsfm, gt_tsfm)

                    rre_list.append(rre.item())
                    rte_list.append(rte.item())
                    tsfm_err_list.append(tsfm_err.item())
                    shape_lst.append(ref['id_label'][ref_ids[0].item()][-1])
                    cd_lst.append(chamfer_l1.item())
                    
                    if visualize:
                        gt_pc_list.append(transform(gt_tsfm, instance_ref))
                        pred_pc_list.append(transform(pred_tsfm, instance_ref))
                        ref_ins_list.append(instance_ref)
                        rescan_ins_list.append(instance_rescan)

                    # debug
                    # print(rre)
                    # vis_src = vis.pointcloud(instance_ref.squeeze()[::5].detach().cpu().numpy(), is_sphere=True, cmap="tab10", ucid=0)
                    # vis_tgt = vis.pointcloud(instance_rescan.squeeze()[::5].detach().cpu().numpy(), is_sphere=True, cmap="tab10", ucid=1)
                    # vis_pred = vis.pointcloud(transform(pred_tsfm, instance_ref).squeeze()[::5].detach().cpu().numpy(), is_sphere=True, cmap="tab10", ucid=0)
                    # vis_gt = vis.pointcloud(transform(gt_tsfm, instance_ref).squeeze()[::5].detach().cpu().numpy(), is_sphere=True, cmap="tab10", ucid=0)
                    # vis.show_3d([vis_src, vis_tgt], [vis_pred, vis_tgt], [vis_gt, vis_tgt],  use_new_api=True, auto_plane=False, up_axis='+Z')

            if visualize:
                try:
                    
                    vis_ref_ins = [vis.pointcloud(pc.squeeze()[::5].detach().cpu().numpy(), is_sphere=True, cmap="tab20", ucid=i*2+1) for i, pc in enumerate(ref_ins_list)]
                    vis_gt_ins = [vis.pointcloud(pc.squeeze()[::5].detach().cpu().numpy(), is_sphere=True, cmap="tab20", ucid=i*2+1) for i, pc in enumerate(gt_pc_list)]
                    vis_pred_ins = [vis.pointcloud(pc.squeeze()[::5].detach().cpu().numpy(), is_sphere=True, cmap="tab20", ucid=i*2+1) for i, pc in enumerate(pred_pc_list)]
                    # pred_rescan_pc = [transform(pred_tsfm, ins_ref) for ins_ref, pred_tsfm in zip(ref_ins_list, pred_pc_list)]
                    
                    # vis target pc
                    rescan_pc = torch.cat(rescan_ins_list, dim=1)
                    vis_rescan_pc = vis.pointcloud(rescan_pc.squeeze()[::5].detach().cpu().numpy(), ucid= 0, is_sphere=True, cmap='tab10')
                    
                    # background point cloud
                    rescan_pc_bg = transform(inverse(scene_tsfm), torch.from_numpy(rescan['bg_pc']).cuda().float()[None])
                    vis_ref_bg = vis.pointcloud(ref['bg_pc'], ucid=8, is_sphere=True, cmap='Pastel1')
                    vis_rescan_bg = vis.pointcloud(rescan_pc_bg.squeeze().detach().cpu().numpy(), ucid=8, is_sphere=True, cmap='Pastel1')
                    # vis_ref_pc = vis.pointcloud(ref_pc.squeeze()[::5].detach().cpu().numpy(), ucid= 0, is_sphere=True, cmap='tab10')
                    
                    # vis_pred_pc = vis.pointcloud(pred_rescan_pc.squeeze()[::5].detach().cpu().numpy(), ucid= 0, is_sphere=True, cmap='tab10')
                    vis.show_3d(vis_ref_ins+[vis_ref_bg], [vis_rescan_pc, vis_rescan_bg], [vis_rescan_pc, vis_rescan_bg] + vis_pred_ins, [vis_rescan_pc, vis_rescan_bg] + vis_gt_ins, use_new_api=True, auto_plane=False, up_axis='+Z')
                # vis.show_3d([vis_rescan_bg, vis_pred_pc, vis_rescan_pc], use_new_api=True, auto_plane=False, up_axis='+Z')
                except: pass

    rre_list = np.array(rre_list)
    rte_list = np.array(rte_list)
    tsfm_err_list = np.array(tsfm_err_list)
    cd_lst = np.array(cd_lst)
    
    rmse_recall = 100*(tsfm_err_list< 0.1).mean()
    med_rre1 = np.median(rre_list[tsfm_err_list<0.2])
    med_rte1 = np.median(rte_list[tsfm_err_list<0.2])
    logging.info(f"Recall [T<0.1m]: {rmse_recall:.2f} | RRE: {med_rre1:.2f} [deg] || RTE: {med_rte1:.2f} [m]")
    
    re_recall = 100*(rre_list<10).mean()
    med_rre2 = np.median(rre_list[rre_list<10])
    med_rte2 = np.median(rte_list[rre_list<10])
    logging.info(f"Recall [RRE<10deg]: {re_recall:.2f}| RRE: {med_rre2:.2f} [deg] || RTE: {med_rte2:.2f} [m]")
    logging.info(f"Chamfer Distance: {np.median(cd_lst):.5f}")

    logging.info(f'------------------------------------')
    os.makedirs(osp.join(solver.cfg['shape_priors']['ckpt_dir'], 'summary'), exist_ok=True)
    

def eval_3rscan_reconstruction(cfg_path, solver, visualize=False):
    data_cfg = load_yaml(cfg_path)
    dataset_3rscan = Dataset_3RScan(data_cfg)
    recon_gt = osp.join(data_cfg['root_path'], "val_set_recon")

    cd_lst, sdf_recall_lst = [], []
    for i, scene in tqdm(enumerate(dataset_3rscan.scene_list)):
        ref_id = scene['reference']
        ref, _ = dataset_3rscan._get_scene(i)

        recon_lst = []
        if ref is None: continue
        for i in range(ref['pc'].shape[0]):
            objectId = ref['objectId'][i]
            gt_mesh = trimesh.load(osp.join(recon_gt, ref_id, f"objectId_{objectId}.ply"))
            with torch.no_grad():
                full_pc = ref['pc'][i]
                pc_mask = ref['pc_mask'][i]
                valid_pc = full_pc.T[pc_mask.squeeze()]
                codes = solver.model.encode_fps(full_pc[None], pc_mask[None])
            # generate mesh
            pred_mesh = solver._mesh_from_latent(codes)

            optim_codes = solver._optimize_code(codes, ref['pc'][i], ref['pc_mask'][i])
            optim_mesh = solver._mesh_from_latent(optim_codes)
            pred_mesh = optim_mesh

            if pred_mesh.vertices.shape[0] != 0:
                cd1, cd2 = compute_chamfer_distance(gt_mesh, pred_mesh, offset=0, scale=1)
                sdf_recall = compute_sdf_recall(pred_mesh, gt_mesh, 0.05)
                cd_lst.append(cd1)
                sdf_recall_lst.append(sdf_recall)
            else:
                sdf_recall_lst.append(0)
            
    logging.info(f"1-way Chamer Distance: {np.mean(cd_lst):.7f}")
    logging.info(f"Mean SDF Recall: {(np.array(sdf_recall_lst)>0.7).mean()*100:.3f}")

if __name__ == '__main__':
    ckpt = "weights"
    dataset_cfg_path = "configs/3rscan.yaml"
    solver_cfg = load_yaml("configs/more_3rscan.yaml")
    
    set_logger(osp.join('eval_3rscan.log'))
    solver_cfg['shape_priors']['ckpt_dir'] = ckpt
    
    logging.info(f'--------Evaluation on 3RScan--------')
    solver = More_Solver(solver_cfg)
    logging.info(f'------------------------------------')
    
    eval_3rscan_matching(dataset_cfg_path, solver)
    evaluate_3rscan_relocalization(dataset_cfg_path, solver)
    eval_3rscan_reconstruction(dataset_cfg_path, solver)

        
