import torch
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

from lib_math.torch_se3 import transform, inverse

from pycg import vis


def solve_R(f1, f2):
    """f1 and f2: (b*)m*3
    only work for batch_size=1
    """
    S = torch.matmul(f1.transpose(-1, -2), f2)  # 3*3
    U, sigma, V = torch.svd(S)
    R = torch.matmul(V, U.transpose(-1, -2))
    det = torch.linalg.det(R)

    diag_1 = torch.tensor([1, 1, 0], device=R.device, dtype=R.dtype)
    diag_2 = torch.tensor([0, 0, 1], device=R.device, dtype=R.dtype)
    det_mat = torch.diag(diag_1 + diag_2 * det)

    det_mat = det_mat.unsqueeze(0)

    R = torch.matmul(V, torch.matmul(det_mat, U.transpose(-1, -2)))
    return R

def kabsch_transformation_estimation(x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0):
    """
    Torch differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimate rotation matrix is then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.

    Args:
        x1            (torch array): points of the first point cloud [b,n,3]
        x2            (torch array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (torch array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]
        normalize_w   (bool)       : flag for normalizing the weights to sum to 1
        best_k        (int)        : number of correspondences with highest weights to be used (if 0 all are used)
        w_threshold   (float)      : only use weights higher than this w_threshold (if 0 all are used)
    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]
        valid_gradient (bool): Flag denoting if the SVD computation converged (gradient is valid)

    """
    if weights is None:
        weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

    if normalize_w:
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = (weights/sum_weights)

    weights = weights.unsqueeze(2)

    if best_k > 0:
        indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
        weights = weights[:,indices,:]
        x1 = x1[:,indices,:]
        x2 = x2[:,indices,:]

    if w_threshold > 0:
        weights[weights < w_threshold] = 0


    x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    weight_matrix = torch.diag_embed(weights.squeeze(2))

    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                           torch.matmul(weight_matrix, x2_centered))

    try:
        u, s, v = torch.svd(cov_mat)
    except Exception as e:
        r = torch.eye(3,device=x1.device)
        r = r.repeat(x1_mean.shape[0],1,1)
        t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

        res = transformation_residuals(x1, x2, r, t)

        return r, t, res, True

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

    # Residuals
    res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res, False


def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters
    
    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t 

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res

def inverse_3d_transform(transform):
    '''
    Inverse 4x4 homogeneous transformation matrix
    Args:
        transform (tensor [B, 4, 4])
    Return: 
        the inverse transform of the input (tensor [B, 4, 4])
    '''
    inverse_tsfm = torch.zeros_like(transform)
    inverse_tsfm[:,3,3] = 1
    R = transform[:, :3, :3]
    t = transform[:, :3, [3]]
    inverse_tsfm[:,:3,:3] = R.transpose(-1, -2)
    inverse_tsfm[:,:3, [3]] = - R.transpose(-1, -2) @ t

    return inverse_tsfm

def solve_transform_from_latent(code1, code2):
    '''
    Solve the relative transform between objects using latent embeddings
    Args: 
        code1, code2 (dict of z_inv, z_so3, s, t)
    Return: 
        4 x 4 matrix  (transform from code1 to code2)
    '''
    R = solve_R(code1['z_so3'], code2['z_so3'])
    t = code2['t'] - torch.einsum("bnm, bjm -> bjn", R, code1['t'])
    tsfm = torch.eye(4).unsqueeze(0).to(R.device)
    tsfm[:, :3, :3] = R
    tsfm[:, :3, [3]] = t.transpose(-1, -2)

    return tsfm


def rotation_error(R1, R2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})

    Args: 
        R1 (torch tensor): Estimated rotation matrices [b,3,3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3]

    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]

    """
    R_ = torch.matmul(R1.transpose(1,2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)

    ae = 180. * ae / torch.pi

    return ae


def translation_error(t1, t2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})

    Args: 
        t1 (torch tensor): Estimated translation vectors [b,3,1]
        t2 (torch tensor): Ground truth translation vectors [b,3,1]

    Returns:
        te (torch tensor): translation error [b,1]

    """
    return torch.norm(t1-t2, dim=(-2,-1))


def evaluate_transform(gt_tsfm, pred_tsfm):
    '''
    Evaluate 3D transformation
    Args: 
        Two [4x4 transformation matrices]
    Return: 
        translation error [deg], rotation error
    '''
    R1, R2 = gt_tsfm[:, :3,:3], pred_tsfm[:, :3,:3]
    t1, t2 = gt_tsfm[:, :3, 3], pred_tsfm[:, :3, 3]
    ae = rotation_error(R1, R2)
    te = translation_error(t1, t2)
    return ae, te


def compute_transformation_error(pc1, pc2, pred_tsfm, gt_tsfm, thres=0.2):
    '''
    Compute transformation error: transform direction pc1 -> pc2
    Args:
        pc1: B, N, 3
        pc2: B, M, 3
        pred_tsfm: B, 4, 4
        gt_tsfm: B, 4, 4
    Return:
        Endpoint RMSE in both ways (1->2 and 2->1)
    '''
    # endpoint error in transforming pc1 -> pc2
    error_12 = transform(pred_tsfm, pc1) - transform(gt_tsfm, pc1)

    # pc2 -> pc1
    error_21 = transform(inverse(pred_tsfm), pc2) - transform(inverse(gt_tsfm), pc2) 

    error = torch.cat([error_12, error_21], dim=1)
    RMSE = (error ** 2).mean().sqrt()
    return RMSE

def visualize_registration(pc_src, pc_tgt, pred_tsfm, gt_tsfm):
    '''
    Visualize registration results in three windows
    Args:
        pc_src: source point cloud (1, N, 3)
        pc_tgt: target point cloud (1, M, 3)
        pred_tsfm: predicted transformation (source -> target)
        gt_tsfm: ground truth transformation (source -> target)
        
    '''
    cmap = 'tab10'
    ucid_src = 0
    ucid_tgt = 1
    vis_pc_src = vis.pointcloud(pc_src.squeeze()[::10].detach().cpu().numpy(), ucid= ucid_src, is_sphere=True, cmap=cmap)
    vis_pc_tgt = vis.pointcloud(pc_tgt.squeeze()[::10].detach().cpu().numpy(), ucid= ucid_tgt, is_sphere=True, cmap=cmap)
    vis_pc_src_tsfm_gt = vis.pointcloud(transform(gt_tsfm, pc_src).squeeze()[::10].detach().cpu().numpy(), ucid= ucid_src, is_sphere=True, cmap=cmap)
    vis_pc_src_tsfm_pred = vis.pointcloud(transform(pred_tsfm, pc_src).squeeze()[::10].detach().cpu().numpy(), ucid= ucid_src, is_sphere=True, cmap=cmap)

    vis.show_3d([vis_pc_src, vis_pc_tgt], [vis_pc_src_tsfm_gt, vis_pc_tgt], [vis_pc_src_tsfm_pred, vis_pc_tgt],use_new_api=True, show=True, up_axis='+Z', point_size=3, auto_plane=True)


def huber_norm_weights(x, b=0.02):
    """
    :param x: norm of residuals, torch.Tensor (N,)
    :param b: threshold
    :return: weight vector torch.Tensor (N, )
    """
    # x is residual norm
    res_norm = torch.zeros_like(x)
    res_norm[x <= b] = x[x <= b] ** 2
    res_norm[x > b] = 2 * b * x[x > b] - b ** 2
    x[x == 0] = 1.
    weight = torch.sqrt(res_norm) / x # = 1 in the window and < 1 out of the window

    return weight

def get_robust_res(res, b):
    """
    :param res: residual vectors
    :param b: threshold
    :return: residuals after applying huber norm
    """
    # print(res.shape[0])
    res = res.view(-1, 1, 1)
    res_norm = torch.abs(res)
    # print(res.shape[0])
    w = huber_norm_weights(res_norm, b=b)
    # print(w.shape[0])
    robust_res = w * res
    # loss = torch.mean(robust_res ** 2) # use l2 loss

    return robust_res, w**2