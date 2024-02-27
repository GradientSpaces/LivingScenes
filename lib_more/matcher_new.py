import torch
from torch import nn
import torch.nn.functional as F
from lib_more.pose_estimation import kabsch_transformation_estimation

'''
Sinkhorn Normalization in Log-Space from SuperGlue
https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
'''

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def sinkhorn_matcher(desc0, desc1, desc_dim=256, match_threshold=0.):
    desc0 = F.normalize(desc0, p=2, dim=1)
    desc1 = F.normalize(desc1, p=2, dim=1)
    scores =  torch.einsum("bdn,bdm->bnm", desc0, desc1)
    scores = scores / desc_dim**.5

    # zan_sinkhorn = sinkhorn(scores.log())
    scores = log_optimal_transport(
            scores, torch.tensor(1.).cuda().float(),
            iters=100)
    # scores = zan_sinkhorn
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    
    return {
            'matches0': indices0.squeeze(), # use -1 for invalid match
            'matches1': indices1.squeeze(), # use -1 for invalid match
    }

def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0]+1)/2, sim_nn.new_tensor(0))
    return matches, scores

def nn_matcher(desc0, desc1):
    desc0 = F.normalize(desc0, p=2, dim=1)
    desc1 = F.normalize(desc1, p=2, dim=1)

    sim = torch.einsum('bdn,bdm->bnm', desc0, desc1)
    
    matches0, scores0 = find_nn(sim, None, None)
    matches1, scores1 = find_nn(sim.permute(0,2,1), None, None)
    matches0 = mutual_check(matches0, matches1)
    matches1 = mutual_check(matches1, matches0)
    return {
            'matches0': matches0.squeeze(), 
            'matches1': matches1.squeeze()
    }

def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new



def sequential_matcher(m0, m1):
    inv_codes_src = F.normalize(m0, p=2, dim=1)
    inv_codes_tgt = F.normalize(m1, p=2, dim=1)
    ids_src = torch.arange(0,len(inv_codes_src), dtype=torch.long).to(inv_codes_src.device)
    ids_tgt = torch.arange(0,len(inv_codes_tgt), dtype=torch.long).to(inv_codes_src.device)
    
    n_iter = min(len(inv_codes_src), len(inv_codes_tgt))
    matched_src = -torch.ones(len(inv_codes_src), dtype=torch.long).to(inv_codes_src.device) # indices of tgt instances in the order of src instances
    matched_tgt = -torch.ones(len(inv_codes_tgt), dtype=torch.long).to(inv_codes_src.device)
    
    # compute the score matching matrix
    score_mat = inv_codes_src @ inv_codes_tgt.T
    
    for i in range(n_iter):
        score_mat = score_mat/(score_mat.max()+1e-5)
        best_pair_index = (score_mat==torch.max(score_mat)).nonzero()
        row_id = best_pair_index[0,0]
        col_id = best_pair_index[0,1]

        # save the match
        matched_src[ids_src[row_id]] = ids_tgt[col_id]
        matched_tgt[ids_tgt[col_id]] = ids_src[row_id]
        ids_src = torch.cat([ids_src[:row_id], ids_src[row_id+1:]])
        ids_tgt = torch.cat([ids_tgt[:col_id], ids_tgt[col_id+1:]])

        # pop
        score_mat = torch.cat([score_mat[:row_id], score_mat[row_id+1:]], dim=0)
        score_mat = torch.cat([score_mat[:,:col_id], score_mat[:,col_id+1:]], dim=1)

    return {'matches0': matched_src,
            'matches1': matched_tgt}


def sim3_seq_matcher(src_codes, tgt_codes):
    # normalize and compute similarity matrix
    inv_codes_src = F.normalize(src_codes['z_inv'].detach().clone(), p=2, dim=1)
    inv_codes_tgt = F.normalize(tgt_codes['z_inv'].detach().clone(), p=2, dim=1)
    sim_mat = inv_codes_src @ inv_codes_tgt.T
    
    # compute the so3 feature residuals
    B, _, _ =  tgt_codes['z_so3'].shape

    res_mat = torch.zeros_like(sim_mat)

    for i, src_so3 in enumerate(src_codes['z_so3']):
        _, _, res, _ = kabsch_transformation_estimation(src_so3[None].repeat_interleave(B, dim=0), tgt_codes['z_so3'])
        res_mat[i] = res.mean(dim=1)

    # combine the score mat and residual matrix
    score_mat = sim_mat / (res_mat+1e-5) 

    ids_src = torch.arange(0,len(inv_codes_src), dtype=torch.long).to(inv_codes_src.device)
    ids_tgt = torch.arange(0,len(inv_codes_tgt), dtype=torch.long).to(inv_codes_src.device)
    
    n_iter = min(len(inv_codes_src), len(inv_codes_tgt))
    matched_src = -torch.ones(len(inv_codes_src), dtype=torch.long).to(inv_codes_src.device) # indices of tgt instances in the order of src instances
    matched_tgt = -torch.ones(len(inv_codes_tgt), dtype=torch.long).to(inv_codes_src.device)
    
    for i in range(n_iter):
        score_mat = score_mat/(score_mat.max()+1e-5)
        best_pair_index = (score_mat==torch.max(score_mat)).nonzero()
        row_id = best_pair_index[:,0]
        col_id = best_pair_index[:,1]

        # save the match
        matched_src[ids_src[row_id]] = ids_tgt[col_id]
        matched_tgt[ids_tgt[col_id]] = ids_src[row_id]
        ids_src = torch.cat([ids_src[:row_id], ids_src[row_id+1:]])
        ids_tgt = torch.cat([ids_tgt[:col_id], ids_tgt[col_id+1:]])

        # pop
        score_mat = torch.cat([score_mat[:row_id], score_mat[row_id+1:]], dim=0)
        score_mat = torch.cat([score_mat[:,:col_id], score_mat[:,col_id+1:]], dim=1)

    return {'matches0': matched_src,
            'matches1': matched_tgt}



def eq_seq_matcher(src_codes, tgt_codes):
    # normalize and compute similarity matrix
    inv_codes_src = F.normalize(src_codes['z_inv'].detach().clone(), p=2, dim=1)
    inv_codes_tgt = F.normalize(tgt_codes['z_inv'].detach().clone(), p=2, dim=1)
    sim_mat = inv_codes_src @ inv_codes_tgt.T
    
    # compute the so3 feature residuals
    B, _, _ =  tgt_codes['z_so3'].shape

    res_mat = torch.zeros_like(sim_mat)

    for i, src_so3 in enumerate(src_codes['z_so3']):
        _, _, res, _ = kabsch_transformation_estimation(src_so3[None].repeat_interleave(B, dim=0), tgt_codes['z_so3'])
        res_mat[i] = res.mean(dim=1)

    # combine the score mat and residual matrix
    score_mat = 1 / (res_mat+1e-5) 

    ids_src = torch.arange(0,len(inv_codes_src), dtype=torch.long).to(inv_codes_src.device)
    ids_tgt = torch.arange(0,len(inv_codes_tgt), dtype=torch.long).to(inv_codes_src.device)
    
    n_iter = min(len(inv_codes_src), len(inv_codes_tgt))
    matched_src = -torch.ones(len(inv_codes_src), dtype=torch.long).to(inv_codes_src.device) # indices of tgt instances in the order of src instances
    matched_tgt = -torch.ones(len(inv_codes_tgt), dtype=torch.long).to(inv_codes_src.device)
    
    for i in range(n_iter):
        score_mat = score_mat/(score_mat.max()+1e-5)
        best_pair_index = (score_mat==torch.max(score_mat)).nonzero()
        row_id = best_pair_index[:,0]
        col_id = best_pair_index[:,1]

        # save the match
        matched_src[ids_src[row_id]] = ids_tgt[col_id]
        matched_tgt[ids_tgt[col_id]] = ids_src[row_id]
        ids_src = torch.cat([ids_src[:row_id], ids_src[row_id+1:]])
        ids_tgt = torch.cat([ids_tgt[:col_id], ids_tgt[col_id+1:]])

        # pop
        score_mat = torch.cat([score_mat[:row_id], score_mat[row_id+1:]], dim=0)
        score_mat = torch.cat([score_mat[:,:col_id], score_mat[:,col_id+1:]], dim=1)

    return {'matches0': matched_src,
            'matches1': matched_tgt}
