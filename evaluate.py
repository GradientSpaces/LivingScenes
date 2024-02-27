import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import point_cloud_utils as pcu
from lib_math import torch_se3
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "./lib_shape_prior/")))
from lib_shape_prior.core.models.utils.occnet_utils.utils.libmesh.inside_mesh import check_mesh_contains

def compute_chamfer_distance(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer, gen_to_gt_chamfer

def compute_volumetric_iou(mesh1, mesh2, voxel_size = 1./16):

    inside_mask = check_mesh_contains(mesh1, mesh2.vertices)
    return inside_mask.mean()

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist

def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def compute_sdf_recall(mesh1, mesh2, thres=0.1):
    '''
    Measure the SDF recall of vertices of mesh2 to mesh1
    '''
    cd, _ = compute_chamfer_distance(mesh1, mesh2, 0, 1)
    query_pts = mesh2.vertices
    sdf, _, _ = pcu.signed_distance_to_mesh(query_pts, mesh1.vertices, mesh1.faces)
    return (np.abs(sdf)<thres).mean()
    
    
    
def chamfer_distance_torch(src, ref, pred_tsfm, gt_tsfm):
    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)
    
    src_transformed = torch_se3.transform(pred_tsfm, src)
    
    ref_inv_transformed = torch_se3.transform(torch_se3.concatenate(pred_tsfm, torch_se3.inverse(gt_tsfm)), ref)
    dist_src = torch.min(square_distance(src_transformed, ref), dim=-1)[0]
    dist_ref = torch.min(square_distance(ref, ref_inv_transformed), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    return chamfer_dist
    
    