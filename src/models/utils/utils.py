import os
import cv2
import numpy as np
import torch
import random

def fix_randomness(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_crop_corr(h, w, crop_center, CROP_SIZE, FINE_FEATURE_MAP_SIZE):
    crop_center = [crop_center[0] * FINE_FEATURE_MAP_SIZE[1], 
                   crop_center[1] * FINE_FEATURE_MAP_SIZE[0]]
    lt_i = max(0, int(crop_center[1]) - CROP_SIZE)
    lt_j = max(0, int(crop_center[0]) - CROP_SIZE)
    rb_i = min(h, int(crop_center[1]) + CROP_SIZE)
    rb_j = min(w, int(crop_center[0]) + CROP_SIZE)

    lt_i = max(0, min(lt_i, rb_i - 2*CROP_SIZE))
    lt_j = max(0, min(lt_j, rb_j - 2*CROP_SIZE))
    rb_i, rb_j = lt_i + 2*CROP_SIZE, lt_j + 2*CROP_SIZE
    rb_i, rb_j = max(rb_i, lt_i + CROP_SIZE), max(rb_j, lt_j + CROP_SIZE)
    return int(lt_i), int(rb_i), int(lt_j), int(rb_j)


def normalize_kpts(kpts, l_or_r):
    kpts = kpts.clone()
    kpts[..., :, 0] = (kpts[..., :, 0])*2 if l_or_r == 'l' else \
                      (kpts[..., :, 0]-0.5)*2
    return kpts


def denormalize_kpts(kpts, l_or_r):
    kpts = kpts.clone()
    kpts[..., :, 0] = (kpts[..., :, 0])/2 if l_or_r == 'l' else \
        (kpts[..., :, 0]/2 + 0.5)
    return kpts


def make_patch_pairs(patch_queries_features, patch_corrs_features):
    patch_pairs = torch.cat([
        patch_queries_features, patch_corrs_features], dim=-1)
    _bk, _c, _w, _ww = patch_pairs.shape
    mask_patch_pairs = torch.full(
        (_bk, _w, _ww), False, device=patch_pairs.device)
       
    return patch_pairs, mask_patch_pairs

def make_query_corrs_pairs(patch_queries, patch_corrs):
    q_c_pairs = torch.cat([patch_queries, patch_corrs], dim=-1)
    return q_c_pairs

def make_pairs(
    patch_queries_features, patch_corrs_features, inner_map, patch_queries, patch_corrs):
    '''
    make patch pairs and assign queries 
    1. feature/mask patches
    2. 
    '''
    patch_pairs = torch.cat(
        [patch_queries_features, patch_corrs_features], dim=-1)
    _bk, _c, _w, _ww = patch_pairs.shape
    mask_patch_pairs = torch.full((_bk, _w, _ww), False, device=patch_pairs.device)
    fine_queries_patches = torch.cat([patch_queries, patch_corrs], dim=-1)
    _bk, _max_q, _d= fine_queries_patches.shape
    if _bk != 0:
        fine_queries_patches = fine_queries_patches.reshape(-1, max(_max_q, 1), _d)
    pad_mask = inner_map[..., 0]
    _mask = torch.sum(pad_mask.float(), dim = 1) > 0
    return (patch_pairs[_mask], mask_patch_pairs[_mask], 
           fine_queries_patches[_mask], pad_mask[_mask])

def get_crop_feat_pos(h,w, crop_center, CROP_SIZE, fine_fmap_size, min_unit=1):
    h,w = h//min_unit, w//min_unit
    fine_fmap_size = fine_fmap_size//min_unit
    CROP_SIZE = CROP_SIZE//min_unit
    
    crop_center = crop_center * fine_fmap_size
    lt_i=max(0, int(crop_center[1]) - CROP_SIZE)
    lt_j=max(0, int(crop_center[0]) - CROP_SIZE)
    rb_i=min(h, int(crop_center[1]) + CROP_SIZE)
    rb_j=min(w, int(crop_center[0]) + CROP_SIZE)

    lt_i = max(0,min(lt_i, rb_i - 2*CROP_SIZE))
    lt_j = max(0,min(lt_j, rb_j - 2*CROP_SIZE))
    rb_i, rb_j = lt_i + 2*CROP_SIZE, lt_j + 2*CROP_SIZE
    return (int(lt_i*min_unit),int(rb_i*min_unit),
            int(lt_j*min_unit),int(rb_j*min_unit))


def show_matches(img0,img1,match_pairs,name,PATCH_H=1024):
    canvas=np.concatenate([img0,img1],axis=1).copy()
    for pair in match_pairs:
        pair[2]+=PATCH_H
        cv2.line(
            canvas,
            (int(pair[0]),int(pair[1])), 
            (int(pair[2]), int(pair[3])), 
            (255,0,0), 2 )
    cv2.imwrite(f'{name}.jpg',canvas)

def draw_matches(img_a, img_b, corrs, type='dot'):
    h0, w0 = img_a.shape[:2]
    h1, w1 = img_b.shape[:2]
    maxh, _ = max(h0, h1), max(w0, w1)
    canvas = np.zeros((maxh, w0+w1, 3)).astype(np.uint8)
    canvas[0:h0, 0:w0] = img_a
    canvas[0:h1, w0:w0+w1] = img_b
    kp_a, kp_b = corrs[:, :2].copy(), corrs[:, 2:4].copy()
    kp_b[:, 0] = kp_b[:, 0] + w0
    if type == 'dot':
        for i in range(len(corrs)):
            value = (kp_a[i, 0]/w0 + (kp_a[i, 1]/h0))/2*255
            cv2.circle(canvas, (round(kp_a[i, 0]), round(kp_a[i, 1])), 
                       2, value_2_jet(value), 2)
            cv2.circle(canvas, (round(kp_b[i, 0]), round(kp_b[i, 1])), 
                       2, value_2_jet(value), 2)
    else :
        for i in range(len(corrs)):
            value = (kp_a[i, 0]/w0 + (kp_a[i, 1]/h0))/2*255
            cv2.line(
            canvas,
            (round(kp_a[i, 0]), round(kp_a[i, 1])),
            (round(kp_b[i, 0]), round(kp_b[i, 1])),
            (255,0,0), 2 )
    return canvas


def value_2_jet(value):
    b, g, r = 0, 0, 0
    if value == 0:
        b, g, r = 0, 0, 0
    elif value <= 51:
        b, g, r = 255, value*5, 0
    elif value < 102:
        value -= 51
        b, g, r = 255-value*5, 255, 0
    elif value <= 153:
        value -= 102
        b, g, r = 0, 255, value*5
    elif value < 204:
        value -= 153
        b, g, r = 0, 255-(128*value/51.0+0.5), 255
    else:
        value -= 204
        b, g, r = 0, 127-(127*value/51+0.5), 255
    return b, g, r
