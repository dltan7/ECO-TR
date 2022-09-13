import pdb
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import sys
import os
SCRIPT_DIR = sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)

from .ecotr_modules.query_clustering import \
    generate_fine_patch_inference_kmeans, map_patch_coord_to_image
from .utils.utils import normalize_kpts, denormalize_kpts, \
    make_patch_pairs, make_query_corrs_pairs
from .ecotr_modules.ecotr_backbone import build_backbone
from .ecotr_modules.transformer import build_transformer
# from .light_transformer import FastCorrespondenceTransformer
from .ecotr_modules.position_encoding import NerfPositionalEncoding, MLP
from .ecotr_modules.misc import (NestedTensor, nested_tensor_from_tensor_list)

class EOTR(nn.Module):
    def __init__(self, backbone, coarse_tsfm, mid_tsfm, fine_tsfm, args):
        super().__init__()
        self.transformer_type= args.transformer_type

        self.backbone = backbone
        self.hidden_dims = backbone.backbone.body.config['dense'][-1]
        self.coarse_transformer = coarse_tsfm
        self.mid_transformer = mid_tsfm
        self.fine_transformer = fine_tsfm

        self.coarse_corr_embed = MLP(
            self.hidden_dims[-1], self.hidden_dims[-1], 2, 3)
        self.coarse_uncertainty_embed = MLP(
            self.hidden_dims[-1], self.hidden_dims[-1], 1, 3) 
        self.coarse_query_proj = NerfPositionalEncoding(
            self.hidden_dims[-1] // 4, args.position_embedding)
        self.coarse_input_proj = nn.Sequential(
                nn.Conv2d(
                    backbone.backbone.body.config['dense'][-1][-1], 
                    self.hidden_dims[-1], kernel_size=1, stride=1, 
                    padding=0, bias=False), 
                nn.GroupNorm(16, self.hidden_dims[-1]),
            )
        
        self.mid_corr_embed = MLP(
            self.hidden_dims[-2], self.hidden_dims[-2], 2, 3)
        self.mid_uncertainty_embed = MLP(
            self.hidden_dims[-2],self.hidden_dims[-2], 1, 3) 
        self.mid_query_proj = NerfPositionalEncoding(
            self.hidden_dims[-2] // 4, args.position_embedding)
        self.mid_input_proj = nn.Sequential(
                nn.Conv2d(
                    backbone.backbone.body.config['dense'][-1][-2], 
                    self.hidden_dims[-2], kernel_size=1, stride=1, 
                    padding=0, bias=False), 
                nn.GroupNorm(16, self.hidden_dims[-2]),
            )
        
        self.fine_corr_embed = MLP(
            self.hidden_dims[0], self.hidden_dims[0], 2, 3)
        self.fine_uncertainty_embed = MLP(
            self.hidden_dims[0], self.hidden_dims[0], 1, 3) 
        self.fine_query_proj = NerfPositionalEncoding(
            self.hidden_dims[0] // 4, args.position_embedding)
        self.fine_input_proj = nn.Sequential(
                nn.Conv2d(
                    backbone.backbone.body.config['dense'][-1][0], 
                    self.hidden_dims[0], kernel_size=1, stride=1, 
                    padding=0, bias=False), 
                nn.GroupNorm(16, self.hidden_dims[0]),
            )
        
        assert args.window_size % 2 == 1, \
        "window_size ({}) should be odd.".format(args.window_size)

        self.num_anchors_per_batch_test_mid = args.num_anchors_per_batch_test_mid
        self.num_anchors_per_batch_test_fine = args.num_anchors_per_batch_test_fine
        self.kmeans_iter_num= args.kmeans_iter_num
        self.window_size = args.window_size
        self.window_size_fine = args.window_size_fine
        self.safe_ratio= args.safe_ratio
        self.minibatchsize = args.minibatchsize 
    
    def forward_coarse(self, data_pack):
        samples = data_pack['samples']
        queries = data_pack['queries']

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples) 
        _, _, coarse_feature = features
        coarse_pos = pos[-1] # [n c H W] , [n c h w]
        
        out = {}
        self.coarse_ecotr(
            coarse_feature, coarse_pos, queries, out)
        return out

    def forward_fine(self,data_pack):
        samples = data_pack['samples']
        queries = data_pack['queries']
        mid_corrs = data_pack['mid_corrs']

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples) 
        fine_feature, _, __ = features
        
        out = {}
        out = self.fine_ecotr(
            fine_feature, queries, mid_corrs, out)
        return out

    def forward(self, data_pack):
        samples = data_pack['samples']
        queries = data_pack['queries']

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        h, w = samples.tensors.shape[-2:]

        features, pos = self.backbone(samples) 
        fine_feature, mid_feature, coarse_feature = features
        coarse_pos = pos[-1] # [n c H W] , [n c h w]
        
        out = {}
        self.coarse_ecotr(
            coarse_feature, coarse_pos, queries, out)
        self.mid_ecotr(
            mid_feature, queries, out['coarse_corrs'], out)
        self.fine_ecotr(
            fine_feature, queries, out['mid_corrs'], out)
        return out

    def coarse_ecotr(self, coarse_feature, coarse_pos, queries, out):
        '''
            Args:
                self, queries
            Outputs:
                coarse_corrs 
        '''
        coarse_src, coarse_mask = coarse_feature.decompose()
        c_h, c_w = coarse_src.shape[-2:]
        assert coarse_mask is not None
        # 
        coarse_queries = queries
        _c_b, _c_q, _ = coarse_queries.shape

        coarse_queries = coarse_queries.reshape(-1, 2)
        coarse_queries = self.coarse_query_proj(coarse_queries)
        coarse_queries = coarse_queries.reshape(_c_b, _c_q, -1)
        coarse_queries = coarse_queries.permute(1, 0, 2) # _c_q, _c_b, 2

        coarse_hs = self.coarse_transformer(
            self.coarse_input_proj(coarse_src), coarse_mask, 
            coarse_queries, coarse_pos)[0]
        
        coarse_corrs = self.coarse_corr_embed(coarse_hs)[-1]
        coarse_corrs_uncertainty = self.coarse_uncertainty_embed(coarse_hs)[-1]
        coarse_corrs_uncertainty = torch.sigmoid(coarse_corrs_uncertainty)
        
        out.update({'coarse_queries':coarse_queries,
                'coarse_corrs':coarse_corrs,
                'coarse_uncertainty': coarse_corrs_uncertainty,
                })
        return 

    def mid_ecotr(self, mid_feature, queries, corrs, out): # val: samples.tensors

        MAX_PARA = self.minibatchsize
        coarse_corrs = corrs
        mid_src, mid_mask = mid_feature.decompose()
        f_h, f_w = mid_src.shape[-2:]
        assert mid_mask is not None
        assert f_h >= self.window_size
        ori_queries = queries.clone()

        out_dict = generate_fine_patch_inference_kmeans(
            mid_src, mid_mask, 
            normalize_kpts(ori_queries,'l'), 
            normalize_kpts(coarse_corrs,'r'),
            window_size=self.window_size, k=self.num_anchors_per_batch_test_mid, 
            kmeans_iter_num=self.kmeans_iter_num,
            safe_ratio=self.safe_ratio
            )

        pad_mask = out_dict['patch_inner_map'][..., 0]
        _mask = torch.sum(pad_mask.float(), dim = 1) > 0
        mid_patch_pairs, mid_mask_patch_pairs = make_patch_pairs(
            out_dict['patch_queries_features'], out_dict['patch_corrs_features'])
        mid_query_corrs_pairs = make_query_corrs_pairs(
            out_dict['patch_queries'],out_dict['patch_corrs'])
        
        mid_patch_pairs = mid_patch_pairs[_mask]
        mid_mask_patch_pairs = mid_mask_patch_pairs[_mask]
        mid_query_corrs_pairs = mid_query_corrs_pairs[_mask]
        pad_mask = pad_mask[_mask]

        if 'alone_chosen_index' in out_dict.keys():
            mid_alone_patch_pairs, mid_alone_mask_patch_pairs = \
                make_patch_pairs(
                    out_dict['alone_queries_features'],
                    out_dict['alone_corrs_features'])
            alone_query_corrs_pairs = make_query_corrs_pairs(
                out_dict['alone_queries'],out_dict['alone_corrs'])

            ### cat [batch, alone]
            mid_patch_pairs = torch.cat([
                mid_patch_pairs, mid_alone_patch_pairs], dim=0)
            mid_mask_patch_pairs = torch.cat([
                mid_mask_patch_pairs, mid_alone_mask_patch_pairs], dim=0)
            max_q = max(mid_query_corrs_pairs.shape[1], 1)
            alone_query_corrs_pairs = alone_query_corrs_pairs.expand(-1, max_q, -1)
            if mid_query_corrs_pairs.shape[0] != 0 and \
                alone_query_corrs_pairs.shape[0] != 0:
                mid_query_corrs_pairs = torch.cat([
                    mid_query_corrs_pairs, alone_query_corrs_pairs], dim=0)
            elif mid_query_corrs_pairs.shape[0] == 0:
                mid_query_corrs_pairs = alone_query_corrs_pairs

            _alone_num = mid_alone_patch_pairs.shape[0]
            alone_pad_mask = torch.cat(
                [out_dict['alone_inner_map'].int().sum(-1) > 0,
                torch.full(
                  (_alone_num, max_q-1), False, device=pad_mask.device)],
                dim=-1)
            
            if pad_mask.shape[0] != 0 and alone_pad_mask.shape[0] != 0:
                pad_mask = torch.cat([pad_mask, alone_pad_mask], dim=0)
            elif pad_mask.shape[0] == 0:
                pad_mask = alone_pad_mask
        
        if mid_patch_pairs is None:
            return None
        
        mid_queries = denormalize_kpts( 
            mid_query_corrs_pairs[...,:2],'l')
        mid_corrs_c = denormalize_kpts(
            mid_query_corrs_pairs[...,2:], 'r')
        
        mid_corrs_list = []
        mid_corrs_uncertainty_list = []
        cnt = np.ceil(mid_patch_pairs.shape[0]/MAX_PARA).astype(int)
        for i in range(cnt):
            _mid_patch_pairs = mid_patch_pairs[MAX_PARA*i:MAX_PARA*(i+1)]
            _mid_mask_patch_pairs = mid_mask_patch_pairs[MAX_PARA*i:MAX_PARA*(i+1)]
            _mid_queries = mid_queries[MAX_PARA*i:MAX_PARA*(i+1)]

            _n,_c,_h,_w = _mid_patch_pairs.shape
            _mid_patch_pairs_nest = NestedTensor(
                _mid_patch_pairs, mask = torch.zeros((_n,_h,_w),
                device = _mid_patch_pairs.device).bool())
            _mid_pos_patches = self.backbone.pos_embeds[1](
                _mid_patch_pairs_nest).to(_mid_patch_pairs_nest.tensors.dtype)
            _c_b, _c_q, _ = _mid_queries.shape

            _mid_queries = _mid_queries.reshape(-1, 2)
            _mid_queries = self.mid_query_proj(_mid_queries).reshape(_c_b, _c_q, -1)
            _mid_queries = _mid_queries.permute(1, 0, 2) # _c_q, _c_b, 2
            _mid_hs = self.mid_transformer(
                self.mid_input_proj(_mid_patch_pairs), 
                _mid_mask_patch_pairs, 
                _mid_queries, 
                _mid_pos_patches)[0]
            _mid_corrs = self.mid_corr_embed(_mid_hs)[-1]
            _mid_corrs_uncertainty = self.mid_uncertainty_embed(_mid_hs)[-1]

            mid_corrs_list.append(_mid_corrs)
            mid_corrs_uncertainty_list.append(_mid_corrs_uncertainty)

        mid_corrs = torch.cat(mid_corrs_list,dim=0)
        mid_corrs_uncertainty = torch.cat(mid_corrs_uncertainty_list,dim=0)

        if 'alone_chosen_index' in out_dict.keys():
            all_queries_centroids = torch.cat([
                out_dict['patch_queries_centroids'], 
                out_dict['alone_queries_centroids']], dim = 0)
            all_corrs_centroids = torch.cat([
                out_dict['patch_corrs_centroids'],
                out_dict['alone_corrs_centroids']], dim = 0)
        else:
            all_queries_centroids = out_dict['patch_queries_centroids']
            all_corrs_centroids = out_dict['patch_corrs_centroids']

        final_queries = map_patch_coord_to_image(
            normalize_kpts(mid_queries,'l'), all_queries_centroids, 
            window_size = self.window_size, 
            image_size = [mid_src.shape[2],mid_src.shape[3]//2])
        final_corrs = map_patch_coord_to_image(
            normalize_kpts(mid_corrs,'r'), all_corrs_centroids,
            window_size = self.window_size, 
            image_size = [mid_src.shape[2],mid_src.shape[3]//2])

        if 'alone_chosen_index' in out_dict.keys():
            patch_idx = out_dict['patch_chosen_index']
            alone_idx = out_dict['alone_chosen_index']
            if patch_idx[...,0].shape[0]!=0 and \
                alone_idx[...,0].shape[0]!=0:
                all_rerank_idx = torch.cat([
                    patch_idx[...,0], 
                    alone_idx[...,0].expand(-1,patch_idx.shape[1])], dim = 0)
            elif patch_idx[...,0].shape[0]==0:
                all_rerank_idx = alone_idx[...,0]
            elif alone_idx[...,0].shape[0]==0:
                all_rerank_idx = patch_idx[...,0]
        else:
            all_rerank_idx = patch_idx[...,0]

        final_queries = final_queries[pad_mask]
        final_corrs = final_corrs[pad_mask]
        all_rerank_idx = all_rerank_idx[pad_mask]
        final_corrs_uncertainty = mid_corrs_uncertainty[pad_mask]

        # rerank by idx
        _idx = all_rerank_idx.argsort()
        final_queries = final_queries[_idx]
        final_corrs = final_corrs[_idx]
        final_corrs_uncertainty = final_corrs_uncertainty[_idx]
        final_queries = denormalize_kpts(final_queries, 'l')
        final_corrs = denormalize_kpts(final_corrs,'r')
        # debug code
        assert(abs(final_queries-queries).max()<=1e-5)

        # Note: only support bz = 1 now!
        final_corrs = final_corrs[None]
        final_corrs_uncertainty = final_corrs_uncertainty[None]
        final_corrs_uncertainty = torch.sigmoid(final_corrs_uncertainty)
        out.update({'mid_corrs': final_corrs,
                    'mid_uncertainty': final_corrs_uncertainty,
        })
        return out 

    def fine_ecotr(self, fine_feature, queries, corrs, out): # val: samples.tensors
        if out == None:
            return None 
        
        MAX_PARA = self.minibatchsize
        coarse_corrs = corrs
        fine_src, fine_mask = fine_feature.decompose()
        f_h, f_w = fine_src.shape[-2:]
        assert fine_mask is not None
        assert f_h >= self.window_size_fine
        ori_queries = queries.clone()

        out_dict = generate_fine_patch_inference_kmeans(
            fine_src, fine_mask, 
            normalize_kpts(ori_queries,'l'), 
            normalize_kpts(coarse_corrs,'r'), \
            window_size=self.window_size_fine, k=self.num_anchors_per_batch_test_fine,
            kmeans_iter_num=self.kmeans_iter_num,
            safe_ratio=self.safe_ratio
            )
        
        pad_mask = out_dict['patch_inner_map'][..., 0]
        _mask = torch.sum(pad_mask.float(), dim = 1) > 0
        fine_patch_pairs, fine_mask_patch_pairs = make_patch_pairs(
            out_dict['patch_queries_features'], out_dict['patch_corrs_features'])
        fine_query_corrs_pairs = make_query_corrs_pairs(
            out_dict['patch_queries'],out_dict['patch_corrs'])
        
        fine_patch_pairs = fine_patch_pairs[_mask]
        fine_mask_patch_pairs = fine_mask_patch_pairs[_mask]
        fine_query_corrs_pairs = fine_query_corrs_pairs[_mask]
        pad_mask = pad_mask[_mask]

        if 'alone_chosen_index' in out_dict.keys():
            fine_alone_patch_pairs, fine_alone_mask_patch_pairs = \
                make_patch_pairs(
                    out_dict['alone_queries_features'],
                    out_dict['alone_corrs_features'])
        
            alone_query_corrs_pairs = make_query_corrs_pairs(
                out_dict['alone_queries'],out_dict['alone_corrs'])

            fine_patch_pairs = torch.cat([
                fine_patch_pairs, fine_alone_patch_pairs], dim=0)
            # 2. mask patch pairs
            fine_mask_patch_pairs = torch.cat([
                fine_mask_patch_pairs, fine_alone_mask_patch_pairs], dim=0)
            # 3. query_corrs_pairs
            max_q = max(fine_query_corrs_pairs.shape[1], 1)
            alone_query_corrs_pairs = alone_query_corrs_pairs.expand(-1, max_q, -1)
            if fine_query_corrs_pairs.shape[0] != 0 and \
                alone_query_corrs_pairs.shape[0] != 0:
                fine_query_corrs_pairs = torch.cat([
                    fine_query_corrs_pairs, alone_query_corrs_pairs], dim=0)
            elif fine_query_corrs_pairs.shape[0] == 0:
                fine_query_corrs_pairs = alone_query_corrs_pairs
            # 4. pad_mask
            _alone_num = fine_alone_patch_pairs.shape[0]
            alone_pad_mask = torch.cat(
                [out_dict['alone_inner_map'].int().sum(-1) > 0,
                torch.full(
                  (_alone_num, max_q-1), False, device=pad_mask.device)],
                dim=-1)

            if pad_mask.shape[0] != 0 and alone_pad_mask.shape[0] != 0:
                pad_mask = torch.cat([pad_mask, alone_pad_mask], dim=0)
            elif pad_mask.shape[0] == 0:
                pad_mask = alone_pad_mask

        if fine_patch_pairs is None:
            return None
        fine_queries = denormalize_kpts(
            fine_query_corrs_pairs[...,:2],'l')
        fine_corrs_c = denormalize_kpts(
            fine_query_corrs_pairs[...,2:], 'r')

        ## 3.2 inference by fine_ecotr
        fine_corrs_list = []
        fine_corrs_uncertainty_list = []
        cnt = np.ceil(fine_patch_pairs.shape[0]/MAX_PARA).astype(int)
        for i in range(cnt):
            _fine_patch_pairs = fine_patch_pairs[MAX_PARA*i:MAX_PARA*(i+1)]
            _fine_mask_patch_pairs = fine_mask_patch_pairs[MAX_PARA*i:MAX_PARA*(i+1)]
            _fine_queries = fine_queries[MAX_PARA*i:MAX_PARA*(i+1)]
            
            _n,_c,_h,_w = _fine_patch_pairs.shape
            _fine_patch_pairs_nest = NestedTensor(
                _fine_patch_pairs,mask=torch.zeros((_n,_h,_w),
                device=_fine_patch_pairs.device).bool())
            _fine_pos_patches = self.backbone.pos_embeds[0](
                _fine_patch_pairs_nest).to(_fine_patch_pairs_nest.tensors.dtype)
            _c_b, _c_q, _ = _fine_queries.shape

            ## fully/linear attention tsfm block
            _fine_queries = _fine_queries.reshape(-1, 2)
            _fine_queries = self.fine_query_proj(_fine_queries).reshape(_c_b, _c_q, -1)
            _fine_queries = _fine_queries.permute(1, 0, 2) # _c_q, _c_b, 2
            _fine_hs = self.fine_transformer(
                self.fine_input_proj(_fine_patch_pairs), 
                _fine_mask_patch_pairs, 
                _fine_queries, 
                _fine_pos_patches)[0]
            _fine_corrs = self.fine_corr_embed(_fine_hs)[-1]
            _fine_corrs_uncertainty = self.fine_uncertainty_embed(_fine_hs)[-1]
            
            fine_corrs_list.append(_fine_corrs)
            fine_corrs_uncertainty_list.append(_fine_corrs_uncertainty)
        
        fine_corrs = torch.cat(fine_corrs_list,dim=0)
        fine_corrs_uncertainty = torch.cat(fine_corrs_uncertainty_list,dim=0)

        if 'alone_chosen_index' in out_dict.keys():
            all_queries_centroids = torch.cat([
                out_dict['patch_queries_centroids'], 
                out_dict['alone_queries_centroids']], dim = 0)
            all_corrs_centroids = torch.cat([
                out_dict['patch_corrs_centroids'], 
                out_dict['alone_corrs_centroids']], dim = 0)
        else:
            all_queries_centroids = out_dict['patch_queries_centroids']
            all_corrs_centroids = out_dict['patch_corrs_centroids']
            
        final_queries = map_patch_coord_to_image(
            normalize_kpts(fine_queries[...,:2],'l'), all_queries_centroids, 
            window_size = self.window_size_fine, 
            image_size = [fine_src.shape[2],fine_src.shape[3]//2])
        final_corrs = map_patch_coord_to_image(
            normalize_kpts(fine_corrs,'r'), all_corrs_centroids,
            window_size = self.window_size_fine, 
            image_size = [fine_src.shape[2],fine_src.shape[3]//2])
        
        if 'alone_chosen_index' in out_dict.keys():
            patch_idx = out_dict['patch_chosen_index']
            alone_idx = out_dict['alone_chosen_index']
            if patch_idx[...,0].shape[0]!=0 and alone_idx[...,0].shape[0]!=0:
                all_rerank_idx = torch.cat([
                    patch_idx[...,0], 
                    alone_idx[...,0].expand(-1,patch_idx.shape[1]) ],dim = 0)
            elif patch_idx[...,0].shape[0]==0:
                all_rerank_idx = alone_idx[...,0]
            elif alone_idx[...,0].shape[0]==0:
                all_rerank_idx = patch_idx[...,0]
        else:
            all_rerank_idx = patch_idx[...,0]

        final_queries = final_queries[pad_mask]
        final_corrs = final_corrs[pad_mask]
        all_rerank_idx = all_rerank_idx[pad_mask]
        final_corrs_uncertainty = fine_corrs_uncertainty[pad_mask]

        # rerank by idx
        _idx = all_rerank_idx.argsort()
        final_queries = final_queries[_idx]
        final_corrs = final_corrs[_idx]
        final_corrs_uncertainty = final_corrs_uncertainty[_idx]
        final_queries = denormalize_kpts(final_queries, 'l')
        final_corrs = denormalize_kpts(final_corrs,'r')
        # debug
        assert(abs(final_queries-queries).max()<=1e-5)

        # Note: only support bz = 1 now! 
        final_corrs = final_corrs[None]
        final_corrs_uncertainty = final_corrs_uncertainty[None]
        final_corrs_uncertainty = torch.sigmoid(final_corrs_uncertainty)
        out.update({'fine_corrs': final_corrs,
                    'fine_uncertainty':final_corrs_uncertainty,
        })
        return out

def build(args):
    backbone = build_backbone(args)
    if args.transformer_type == 'fully':
        coarse_transformer = build_transformer(args, \
            backbone.backbone.body.config['dense'][-1][-1], 
            backbone.backbone.body.config['dense'][-1][-1] * 2)
        mid_transformer = build_transformer(args, \
            backbone.backbone.body.config['dense'][-1][-2], 
            backbone.backbone.body.config['dense'][-1][-2] * 2)
        fine_transformer = build_transformer(args, \
            backbone.backbone.body.config['dense'][-1][0], 
            backbone.backbone.body.config['dense'][-1][0] * 2)

    model = EOTR(
            backbone,
            coarse_transformer,
            mid_transformer,
            fine_transformer,
            args
        )    
    return model
