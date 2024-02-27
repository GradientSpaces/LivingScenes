import os, json, yaml
from pycg import vis, image, exp

def load_json(filename):
    file = open(filename)
    data = json.load(file)
    file.close()
    return data

def load_yaml(filename):
    file = open(filename)
    data = yaml.safe_load(file)
    file.close()
    return data

def read_list_from_txt(filename):
    file = open(filename, "r")
    return [line.rstrip() for line in file]

def visualize_shape_matching(ref, rescan, matched_ref_ids, rescan_ids):
    '''
    Visualize the shape matching between two point clouds
    '''
    # defining colormap
    fg_cmap, bg_cmap, matched_cmap = "tab10", "Pastel1", "Paired"
    ref_pc = [ref['pc'][i] for i in range(ref["pc"].shape[0])]
    ref_vis = []
    for i, pc in enumerate(ref_pc):
        # if ref_ids.view(-1)[i] != -1:
        ref_vis.append(vis.pointcloud(ref_pc[i].squeeze().detach().cpu().numpy().T[::10], ucid=i, is_sphere=True, sphere_radius=0.03, cmap=fg_cmap))
        
    ref_vis.append(vis.pointcloud(ref['bg_pc'], ucid=8, is_sphere=True, sphere_radius=0.03, cmap=bg_cmap))

    ref_vis_matched = []
    for i, pc in enumerate(ref_pc):
        if matched_ref_ids.view(-1)[i] != -1:
            ref_vis_matched.append(vis.pointcloud(pc.squeeze().detach().cpu().numpy().T[::10], ucid=matched_ref_ids.view(-1)[i].item(), is_sphere=True, sphere_radius=0.03, cmap=matched_cmap))
        else: 
            ref_vis_matched.append(vis.pointcloud(pc.squeeze().detach().cpu().numpy().T[::10], ucid=10, is_sphere=True, sphere_radius=0.03, cmap=bg_cmap))
    ref_vis_matched.append(vis.pointcloud(ref['bg_pc'], ucid=8, is_sphere=True, sphere_radius=0.03, cmap=bg_cmap))

    rescan_pc = [rescan['pc'][i] for i in range(rescan["pc"].shape[0])]
    rescan_vis = []
    for i, pc in enumerate(rescan_pc):
        if rescan_ids.view(-1)[i] != -1:
            rescan_vis.append(vis.pointcloud(pc.squeeze().detach().cpu().numpy().T[::10], ucid=rescan_ids.view(-1)[i].item(), is_sphere=True, sphere_radius=0.03, cmap=matched_cmap))
        else:
            rescan_vis.append(vis.pointcloud(pc.squeeze().detach().cpu().numpy().T[::10], ucid=10, is_sphere=True, sphere_radius=0.03, cmap=bg_cmap))
    rescan_vis.append(vis.pointcloud(rescan['bg_pc'], ucid=8, is_sphere=True, sphere_radius=0.03, cmap=bg_cmap))
    
    ref_gt_vis = []
    for i, pc in enumerate(ref_pc):
        if matched_ref_ids.view(-1)[i] != -1:
            ref_gt_vis.append(vis.pointcloud(pc.squeeze().detach().cpu().numpy().T[::10], ucid=ref['objectId'][i], is_sphere=True, sphere_radius=0.03, cmap=matched_cmap))
    ref_gt_vis.append(vis.pointcloud(ref['bg_pc'], ucid=8, is_sphere=True, sphere_radius=0.03, cmap=bg_cmap))


    vis.show_3d(ref_vis, rescan_vis, ref_vis_matched, ref_gt_vis, use_new_api=True, show=True, up_axis='+Z')