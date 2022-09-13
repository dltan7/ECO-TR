import pdb
import torch
import torch.nn.functional as F 
from einops import rearrange
#from pykeops.torch import LazyTensor

EPS = 1e-6 # avoid nan
@torch.no_grad()
def KMeans(x, K=10, Niter=100):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    B, N, D = x.shape  # Number of samples, dimension of the ambient space
    c = x[:, :K, :].clone()  # Simplistic initialization for the centroids

    # x_i = LazyTensor(x.view(B, N, 1, D))  # (B, N, 1, D) samples
    # c_j = LazyTensor(c.view(B, 1, K, D))  # (B, 1, K, D) centroids
    x_i = x.view(B, N, 1, D).clone()  # (B, N, 1, D) samples
    c_j = c.view(B, 1, K, D).clone()  # (B, 1, K, D) centroids

    # K-means loop:
    # - x  is the (B, N, D) point cloud,
    # - cl is the (B, N,) vector of class labels
    # - c  is the (B, K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (B, N, K) symbolic squared distances
        cl = D_ij.argmin(dim=2).long().squeeze(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(1, cl[..., None].repeat(1, 1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.zeros(B, K, dtype=cl.dtype, device=cl.device) # (B, K)
        Ncl.scatter_add_(1, cl, torch.ones_like(cl))
        Ncl[Ncl==0] = 1
        c /= Ncl.type_as(c).unsqueeze(-1)  # in-place division to compute the average

    return cl, c

#TODO: to be polished
@torch.no_grad()
def generate_fine_patch_inference_kmeans(
        features, masks, 
        queries, corrs, 
        window_size, 
        k=128, 
        kmeans_iter_num=10,
        safe_ratio=0.8):
    
    b, c, h, Dw = features.size()
    w = Dw // 2
    wh_ratio = float(w) / h
    q = queries.size(1)
    assert b == queries.size(0)
    device = features.device

    assert torch.sum(queries >= 1) == 0 and torch.sum(queries < 0) == 0
    corrs[corrs >= 1] = 1 - EPS
    corrs[corrs <  0] = 0

    queries[...,0] *= wh_ratio
    corrs[...,0] *= wh_ratio

    k=min(k,q)
    anchor_splits, anchors = KMeans(queries, k, kmeans_iter_num) 
    queries_distance = torch.cdist(queries, anchors, p=2.0) # Tensor [b,q,k]
    inner_map = queries_distance < (window_size / h / 2.0 * safe_ratio) 
    max_q = min(5000, int(torch.max(torch.sum(inner_map.float(), dim=1)).item()))

    # from queries side get patch-form queries and corrs
    patch_queries = torch.zeros((b, k, max_q, 2), device=device) # [b, k, max_q, 2]
    patch_corrs = torch.zeros((b, k, max_q, 2), device=device) # [b, k, max_q, 2]
    patch_chosen_index = torch.zeros((b, k, max_q, 2), device=device) # [b, k, max_q, 2]: [0, q]
    first_round_chosen_map = torch.sum(inner_map.float(), dim=2) > 0 # [b, q]
    for kk in range(k):
        kk_chosen_map = torch.logical_and(anchor_splits == kk, first_round_chosen_map) # [b, q]
        kk_chosen_map_v, kk_chosen_map_i = torch.sort(kk_chosen_map.float(), dim=1, descending=True) # [b, q]
        kk_chosen_map_i = kk_chosen_map_i[:, :max_q].unsqueeze(1).expand(-1, 2, -1).transpose(1,2) # [b, max_q, 2]
        kk_chosen_map_v = kk_chosen_map_v[:, :max_q, None].float()
        patch_chosen_index[:, kk, :, :] = kk_chosen_map_i # [b, max_q, 2]
        patch_queries[:, kk, :, :] = torch.gather(
            queries, dim=1, index=kk_chosen_map_i) * kk_chosen_map_v # Tensor [b, max_q, 2]
        patch_corrs[:, kk, :, :] = torch.gather(
            corrs, dim=1, index=kk_chosen_map_i) * kk_chosen_map_v # Tensor [b, max_q, 2]
    
    # get corrs-side outsider mask
    corrs_centroids = torch.sum(patch_corrs, dim=2) / \
        (torch.sum((torch.sum(patch_corrs, dim=-1) > 0).float(), dim=2, keepdim=True) + EPS) # [b, k, 2]
    corrs_distance = patch_corrs - corrs_centroids.unsqueeze(2).expand_as(patch_corrs) # [b, k, max_q, 2]
    corrs_distance = torch.sqrt(torch.sum(torch.square(corrs_distance), dim=-1)) # [b, k, max_q]
    corrs_outer_map = corrs_distance >= (window_size / h / 2.0 * safe_ratio) # [b, k, max_q]
    patch_outer_map = torch.logical_or(corrs_outer_map, torch.sum(patch_queries, dim=-1) <= 0)
    patch_outer_map = patch_outer_map.unsqueeze(-1).expand_as(patch_queries)
    patch_inner_map = torch.logical_not(patch_outer_map)

    # Calibrate centroids slightly
    patch_queries_centroids = (anchors * h).long() / h
    patch_corrs_centroids = (anchors * h).long() / h
    
    # normalize patch queries and corrs and mask out outsiders
    patch_queries = (patch_queries - patch_queries_centroids.unsqueeze(2).expand_as(patch_queries)) / (window_size / h) + 0.5
    patch_corrs = (patch_corrs - patch_corrs_centroids.unsqueeze(2).expand_as(patch_corrs)) / (window_size / h) + 0.5

    patch_queries[patch_outer_map] = 0.0 # [b, k, max_q, 2]
    patch_corrs[patch_outer_map] = 0.0 # [b, k, max_q, 2]
    patch_chosen_index[patch_outer_map] = -1 # [b, k, max_q, 2]: [-1, q], `-1` for outliner

    # convert 2d [0, 1] centroids to feature size [0, h*h] 1d index
    patch_queries_centroids_idx = (patch_queries_centroids * h).long()
    patch_queries_centroids_idx = patch_queries_centroids_idx[..., 1] * w + patch_queries_centroids_idx[..., 0]
    patch_corrs_centroids_idx = (patch_corrs_centroids * h).long()
    patch_corrs_centroids_idx = patch_corrs_centroids_idx[..., 1] * w + patch_corrs_centroids_idx[..., 0] # [b, k]

    # generate image patches
    queries_features, corrs_features = features[..., :w], features[..., w:]
    queries_features_unfold = F.unfold(queries_features, kernel_size=window_size, stride=1, padding=window_size//2) # [b, (cww) l]
    queries_features_unfold = queries_features_unfold.transpose(1,2) # [b, l, (cww)]
    patch_queries_centroids_idx_cww = patch_queries_centroids_idx.unsqueeze(1).expand(-1, queries_features_unfold.size(-1), -1).transpose(1,2) # [b, k (cww)]
    patch_queries_features = torch.gather(queries_features_unfold, dim=1, index=patch_queries_centroids_idx_cww) # [b, k, (cww)]
    patch_queries_features = rearrange(patch_queries_features, 'n l (c ww) -> n l c ww', ww=window_size**2) # [b, k, c, ww]
    patch_queries_features = patch_queries_features.reshape(b, k, c, window_size, window_size) # [b, k, c, w, w]

    corrs_features_unfold = F.unfold(corrs_features, kernel_size=window_size, stride=1, padding=window_size//2) # [b, (cww) l]
    corrs_features_unfold = corrs_features_unfold.transpose(1,2) # [b, l, (cww)]
    patch_corrs_centroids_idx_cww = patch_corrs_centroids_idx.unsqueeze(1).expand(-1, corrs_features_unfold.size(-1), -1).transpose(1,2) # [b, k (cww)]
    patch_corrs_features = torch.gather(corrs_features_unfold, dim=1, index=patch_corrs_centroids_idx_cww) # [b, k, (cww)]
    patch_corrs_features = rearrange(patch_corrs_features, 'n l (c ww) -> n l c ww', ww=window_size**2) # [b, k, c, ww]
    patch_corrs_features = patch_corrs_features.reshape(b, k, c, window_size, window_size) # [b, k, c, w, w]

    # from queries side get alone-form queries 
    second_round_chosen_map = torch.full((b, q), False, device=device) # [b, q]
    for _b in range(b):
        _b_chosen_index = patch_chosen_index[_b, :, :, 0][patch_inner_map[_b, :, :, 0]] # [k, max_q] -> [m]
        second_round_chosen_map[_b, :][torch.unique(_b_chosen_index).long()] = True

    patch_q_cnt = torch.sum(patch_inner_map[...,0].float(), dim=-1)  # [b, k, max_q] -> [b, k]
    patch_nonzero_idxs = (patch_q_cnt > 0).nonzero(as_tuple=False) # [s, 2]
    patch_nonzero_idxs = patch_nonzero_idxs[:,0] * k + patch_nonzero_idxs[:,1] # [s]
    patch_queries_features = rearrange(patch_queries_features, 'b k c w1 w2 -> (b k) c w1 w2')[patch_nonzero_idxs] # [s, c, w, w]
    patch_corrs_features = rearrange(patch_corrs_features, 'b k c w1 w2 -> (b k) c w1 w2')[patch_nonzero_idxs] # [s, c, w, w]
    patch_inner_map = rearrange(patch_inner_map, 'b k q r -> (b k) q r')[patch_nonzero_idxs] # [s, max_q, 2]
    patch_queries = rearrange(patch_queries, 'b k q r -> (b k) q r')[patch_nonzero_idxs] # [s, max_q, 2]
    patch_corrs = rearrange(patch_corrs, 'b k q r -> (b k) q r')[patch_nonzero_idxs] # [s, max_q, 2]
    patch_queries_centroids = rearrange(patch_queries_centroids, 'b k r -> (b k) r')[patch_nonzero_idxs] # [s, 2]
    patch_corrs_centroids = rearrange(patch_corrs_centroids, 'b k r -> (b k) r')[patch_nonzero_idxs] # [s, 2]
    patch_chosen_index = rearrange(patch_chosen_index, 'b k q r -> (b k) q r')[patch_nonzero_idxs] # [s, max_q, 2]
    patch_masks = torch.full((b, h, Dw), False, device=device)

    # alone queries 
    alone_map = torch.logical_not(second_round_chosen_map) # [b, q]
    max_alone_q = int(torch.max(torch.sum(alone_map.float(), dim=1)).item())

    alone_map_v, alone_map_i = torch.sort(alone_map.float(), dim=1, descending=True) # [b, q]
    alone_map_i = alone_map_i[:, :max_alone_q].unsqueeze(1).expand(-1, 2, -1).transpose(1,2) # [b, max_alone_q, 2]
    alone_map_v = alone_map_v[:, :max_alone_q, None].float()
    alone_chosen_index = alone_map_i.unsqueeze(2).clone() # [b, max_alone_q, 1, 2]: [0, q]
    alone_queries = (torch.gather(queries, dim=1, index=alone_map_i) * alone_map_v).unsqueeze(2) # Tensor [b, max_alone_q, 1, 2]
    alone_corrs = (torch.gather(corrs, dim=1, index=alone_map_i) * alone_map_v).unsqueeze(2) # Tensor [b, max_alone_q, 1, 2]

    alone_outer_map = (torch.sum(alone_queries, dim=-1, keepdim=True) < 0).expand_as(alone_queries)
    alone_inner_map = torch.logical_not(alone_outer_map)

    # Calibrate centroids slightly
    alone_queries_centroids = (alone_queries * h).long() / h
    alone_corrs_centroids = (alone_corrs * h).long() / h
    
    # normalize patch queries and corrs and mask out outsiders
    alone_queries = (alone_queries - alone_queries_centroids) / (window_size / h) + 0.5
    alone_corrs = (alone_corrs - alone_corrs_centroids) / (window_size / h) + 0.5

    alone_queries_centroids = alone_queries_centroids.squeeze(2)
    alone_corrs_centroids = alone_corrs_centroids.squeeze(2)
    alone_queries[alone_outer_map] = 0.0 # [b, max_alone_q, 1, 2]
    alone_corrs[alone_outer_map] = 0.0 # [b, max_alone_q, 1, 2]
    alone_chosen_index[alone_outer_map] = -1 # [b, max_alone_q, 1, 2]: [-1, q], `-1` for outliner

    # convert 2d [0, 1] centroids to feature size [0, h*h] 1d index
    alone_queries_centroids_idx = (alone_queries_centroids * h).long()
    alone_queries_centroids_idx = alone_queries_centroids_idx[..., 1] * w + alone_queries_centroids_idx[..., 0]
    alone_corrs_centroids_idx = (alone_corrs_centroids * h).long()
    alone_corrs_centroids_idx = alone_corrs_centroids_idx[..., 1] * w + alone_corrs_centroids_idx[..., 0] # [b, max_alone_q]

    alone_queries_centroids_idx_cww = alone_queries_centroids_idx.unsqueeze(1).expand(-1, queries_features_unfold.size(-1), -1).transpose(1,2) # [b, max_alone_q (cww)]
    alone_queries_features = torch.gather(queries_features_unfold, dim=1, index=alone_queries_centroids_idx_cww) # [b, max_alone_q, (cww)]
    alone_queries_features = rearrange(alone_queries_features, 'n l (c ww) -> n l c ww', ww=window_size**2) # [b, max_alone_q, c, ww]
    alone_queries_features = alone_queries_features.reshape(b, max_alone_q, c, window_size, window_size) # [b, max_alone_q, c, w, w]

    alone_corrs_centroids_idx_cww = alone_corrs_centroids_idx.unsqueeze(1).expand(-1, corrs_features_unfold.size(-1), -1).transpose(1,2) # [b, max_alone_q (cww)]
    alone_corrs_features = torch.gather(corrs_features_unfold, dim=1, index=alone_corrs_centroids_idx_cww) # [b, max_alone_q, (cww)]
    alone_corrs_features = rearrange(alone_corrs_features, 'n l (c ww) -> n l c ww', ww=window_size**2) # [b, max_alone_q, c, ww]
    alone_corrs_features = alone_corrs_features.reshape(b, max_alone_q, c, window_size, window_size) # [b, max_alone_q, c, w, w]

    alone_q_cnt = alone_inner_map[...,0 ,0].float()  # [b, max_alone_q(k)] 
    alone_nonzero_idxs = (alone_q_cnt > 0).nonzero(as_tuple=False) # [s, 2]
    alone_nonzero_idxs = alone_nonzero_idxs[:,0] * max_alone_q + alone_nonzero_idxs[:,1] # [s]
    alone_queries_features = rearrange(alone_queries_features, 'b k c w1 w2 -> (b k) c w1 w2')[alone_nonzero_idxs] # [s, c, w, w]
    alone_corrs_features = rearrange(alone_corrs_features, 'b k c w1 w2 -> (b k) c w1 w2')[alone_nonzero_idxs] # [s, c, w, w]
    alone_inner_map = rearrange(alone_inner_map, 'b k q r -> (b k) q r')[alone_nonzero_idxs] # [s, 1, 2]
    alone_queries = rearrange(alone_queries, 'b k q r -> (b k) q r')[alone_nonzero_idxs] # [s, 1, 2]
    alone_corrs = rearrange(alone_corrs, 'b k q r -> (b k) q r')[alone_nonzero_idxs] # [s, 1, 2]
    alone_queries_centroids = rearrange(alone_queries_centroids, 'b k r -> (b k) r')[alone_nonzero_idxs] # [s, 2]
    alone_corrs_centroids = rearrange(alone_corrs_centroids, 'b k r -> (b k) r')[alone_nonzero_idxs] # [s, 2]
    alone_chosen_index = rearrange(alone_chosen_index, 'b k q r -> (b k) q r')[alone_nonzero_idxs] # [s, 1, 2]

    patch_queries_centroids[...,0] = patch_queries_centroids[...,0] / wh_ratio
    patch_corrs_centroids[...,0] = patch_corrs_centroids[...,0] / wh_ratio
    alone_queries_centroids[...,0] = alone_queries_centroids[...,0] / wh_ratio
    alone_corrs_centroids[...,0] = alone_corrs_centroids[...,0] / wh_ratio

    out_pack = {
        'patch_queries_features': patch_queries_features, #[all patch num, c, w, w] 
        'patch_masks': patch_masks, # [1, 52, 104]
        'patch_corrs_features': patch_corrs_features, 
        'patch_inner_map': patch_inner_map, # [191, 68, 2]
        'patch_queries': patch_queries, #[191, 68, 2]
        'patch_corrs': patch_corrs, 
        'patch_queries_centroids': patch_queries_centroids, #[191, 2]
        'patch_corrs_centroids': patch_corrs_centroids,
        'patch_chosen_index': patch_chosen_index, # [191, 68, 2]

        'alone_queries_features': alone_queries_features,  #[all single patch num, c, w, w]
        'alone_corrs_features': alone_corrs_features,
        'alone_inner_map': alone_inner_map, # [33, 1, 2]
        'alone_queries': alone_queries, # [33, 1, 2]
        'alone_corrs': alone_corrs,
        'alone_queries_centroids': alone_queries_centroids, # [33,2]
        'alone_corrs_centroids': alone_corrs_centroids,
        'alone_chosen_index': alone_chosen_index, # [33,1,2]
    }

    return out_pack
@torch.no_grad()
def map_patch_coord_to_image(patch_coords, patch_coords_centroids, window_size, image_size): 
    image_coords_x = (patch_coords[...,0] - 0.5) * window_size / image_size[1] + \
                      patch_coords_centroids[...,0].unsqueeze(1).expand_as(patch_coords[...,0])
    image_coords_y = (patch_coords[...,1] - 0.5) * window_size / image_size[0] + \
                      patch_coords_centroids[...,1].unsqueeze(1).expand_as(patch_coords[...,1])
    image_coords = torch.stack([image_coords_x, image_coords_y],dim=-1)
    return image_coords

