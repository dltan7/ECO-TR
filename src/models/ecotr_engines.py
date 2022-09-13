import pdb
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as tvtf

from src.models.ecotr_model import build
from src.models.utils.utils import normalize_kpts
from src.models.utils.inference_helper import two_images_side_by_side, \
    get_corner, crop_image, clip_points, get_scales

class ECOTR_Engine(torch.nn.Module):
    def __init__(self, opt):
        super(ECOTR_Engine, self).__init__()
        self.opt = opt
        self.ASPECT_RATIOS = opt.engine.aspect_ratios
        self.MAX_LEN = opt.engine.max_len
        self.MAX_KPTS_NUM = opt.engine.max_kpts_num
        self.CYCLE_THRESH = opt.engine.cycle_thresh
        self.coarse_only = False
        self.device = opt.engine.device

        self.model = build(opt)
        self.load_weight(device=self.device)

    def load_weight(self,device='cpu'):
        weights = torch.load(self.opt.load_weights_path, map_location=device)
        self.model.load_state_dict(weights)
        self.model.to(device)
        # utils.safe_load_weights(self.model, weights)

    def forward(self, img_a, img_b, queries=None, cycle = False, level='fine'):
        h_b, w_b = img_b.shape[:2]
        if queries is None:
            queries = self.get_queries(img_a)
        out = self.forward_queries(img_a, img_b, queries)
        corrs = out[f'{level}_corrs']
        unc = out['fine_unc'] + out['mid_unc'] + out['coarse_unc']
        clip_points(corrs, h_b, w_b)
        res = np.concatenate([
            queries, corrs, unc], axis=1)
        
        if cycle:
            out2 = self.forward_queries(img_b, img_a, corrs)
            cycle_mask_c = np.linalg.norm(
                queries-out2['coarse_corrs'],axis=1) < self.CYCLE_THRESH[0]
            cycle_mask_m = np.linalg.norm(
                queries-out2['mid_corrs'],axis=1) < self.CYCLE_THRESH[1]
            cycle_mask_f = np.linalg.norm(
                queries-out2['fine_corrs'],axis=1) < self.CYCLE_THRESH[2]
            cycle_mask = np.logical_and( 
                np.logical_and(cycle_mask_c,cycle_mask_m),cycle_mask_f)
            res = res[cycle_mask]
        return res

    def forward_coarse(self, img_a, img_b, queries=None, cycle = True):
        h_b, w_b = img_b.shape[:2]
        if queries is None:
            queries = self.get_queries(img_a)
        self.coarse_only=True
        out = self.forward_queries(img_a, img_b, queries)
        self.coarse_only=False
        corrs = out['coarse_corrs']
        clip_points(corrs, h_b, w_b)
        res = np.concatenate([
            queries, corrs], axis=1)

        if cycle:
            out2 = self.forward_queries(img_b, img_a, corrs)
            cycle_mask = np.linalg.norm(
                queries-out2['coarse_corrs'],axis=1)<self.CYCLE_THRESH[0]
            res = res[cycle_mask]
        return res

    @torch.no_grad()
    def forward_queries(self, img_a, img_b, queries):
        MAX_LEN = self.MAX_LEN
        ASPECT_RATIOS = self.ASPECT_RATIOS
        hw_a=np.array(img_a.shape[:2])
        hw_b=np.array(img_b.shape[:2])

        hw_new_a, ori_scale_a, scale_a = get_scales(MAX_LEN, ASPECT_RATIOS, hw_a)
        hw_new_b, ori_scale_b, scale_b = get_scales(MAX_LEN, ASPECT_RATIOS, hw_b)
        img_a = cv2.resize(img_a, tuple(hw_new_a[::-1].tolist()))
        img_b = cv2.resize(img_b, tuple(hw_new_b[::-1].tolist()))

        queries = queries * ori_scale_a[::-1] * scale_a[::-1]  # ori -> M -> 1
        queries[:, 0] /= 2

        img = two_images_side_by_side(img_a, img_b)
        rgb_mean, rgb_var = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img = tvtf.normalize(tvtf.to_tensor(img), rgb_mean, rgb_var).float()
        img = img.to(self.device)
        queries = torch.from_numpy(queries).float()
        if self.coarse_only:
            out = self.model.forward_coarse({
                'samples': img[None],
                'queries': queries[None].to(img.device)})
        else:
            out = self.model.forward({
                'samples': img[None],
                'queries': queries[None].to(img.device)})
        
        dict = {}
        if 'coarse_corrs' in out.keys():
            coarse = out['coarse_corrs'][0]
            coarse = normalize_kpts(coarse,'r')
            coarse = coarse.cpu().numpy()
            coarse = coarse / scale_b[::-1]/ori_scale_b[::-1]
            dict.update({'coarse_corrs':coarse})
            unc = out['coarse_uncertainty'][0]
            unc = unc.cpu().numpy()
            dict.update({'coarse_unc':unc})

        if 'mid_corrs' in out.keys():
            mid = out['mid_corrs'][0]
            mid = normalize_kpts(mid,'r')
            mid = mid.cpu().numpy()
            mid = mid / scale_b[::-1] / ori_scale_b[::-1]
            dict.update({'mid_corrs':mid})
            unc = out['mid_uncertainty'][0]
            unc = unc.cpu().numpy()
            dict.update({'mid_unc':unc})

        if 'fine_corrs' in out.keys():
            fine = out['fine_corrs'][0]
            fine = normalize_kpts(fine,'r')
            fine = fine.cpu().numpy()
            fine = fine / scale_b[::-1] / ori_scale_b[::-1]
            dict.update({'fine_corrs':fine})
            unc = out['fine_uncertainty'][0]
            unc = unc.cpu().numpy()
            dict.update({'fine_unc':unc})

        return dict

    def get_queries(self, img_a):
        h, w, _ = img_a.shape
        mask_valid = torch.ones(h, w)
        valid_kpts = torch.where(mask_valid > 0)
        all_queries = torch.stack(valid_kpts, dim=1)
        MAX_KPTS_NUM = self.MAX_KPTS_NUM 
        SAMPLE_RATE_X = int(w / np.sqrt(MAX_KPTS_NUM))
        SAMPLE_RATE_Y = int(h / np.sqrt(MAX_KPTS_NUM))
        sampled_mask = torch.logical_and(
            (all_queries[:, 0] % SAMPLE_RATE_X == int(SAMPLE_RATE_X / 2)),
            (all_queries[:, 1] % SAMPLE_RATE_Y == int(SAMPLE_RATE_Y / 2)))
        queries = all_queries[sampled_mask]
        queries = queries.float().cpu().numpy()[..., [1, 0]].copy()
        return queries

    @torch.no_grad()
    def forward_refine(self, img_a, img_b, queries, corrs):
        '''
        Input: matches produced by keypoint based method
        Output: refined matches
        '''
        MAX_LEN = self.MAX_LEN
        ASPECT_RATIOS = self.ASPECT_RATIOS
        h0, w0, _ = img_a.shape
        h1, w1, _ = img_b.shape

        if len(queries) == 0:
            queries = np.stack([[w0/3, h0/3], [w0*2/3, h0*2/3]], axis=0)
            corrs = np.stack([[w1/3, h1/3], [w1*2/3, h1*2/3]], axis=0)
        q_board = get_corner(queries)
        t_board = get_corner(corrs)
        img_a, q_base_coors = crop_image(img_a, q_board)
        img_b, t_base_coors = crop_image(img_b, t_board)        
        queries = queries - np.array(q_base_coors)
        corrs = corrs - np.array(t_base_coors)

        cropped_hw_a=np.array(img_a.shape[:2])
        cropped_hw_b=np.array(img_b.shape[:2])
        hw_new_a, ori_scale_a, scale_a = get_scales(
            MAX_LEN, ASPECT_RATIOS, cropped_hw_a)
        hw_new_b, ori_scale_b, scale_b = get_scales(
            MAX_LEN, ASPECT_RATIOS, cropped_hw_b)

        img_a = cv2.resize(img_a, tuple(hw_new_a[::-1].tolist()))
        img_b = cv2.resize(img_b, tuple(hw_new_b[::-1].tolist()))

        queries = queries * ori_scale_a[::-1] * scale_a[::-1]  # ori -> M -> 1
        queries[:, 0] /= 2
        corrs = corrs * ori_scale_b[::-1] * scale_b[::-1]  # ori -> M -> 1
        corrs[:, 0] = corrs[:, 0]/2. + 0.5

        img = two_images_side_by_side(img_a, img_b)
        rgb_mean, rgb_var = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img = tvtf.normalize(tvtf.to_tensor(img), rgb_mean, rgb_var).float()
        img = img.to(self.device)
        queries = torch.from_numpy(queries).float()
        corrs = torch.from_numpy(corrs).float()

        out = self.model.forward_fine({
            'samples': img[None], 
            'queries': queries[None].to(img.device),
            'mid_corrs': corrs[None].to(img.device)
            })

        fine = out['fine_corrs']
        unc = out['fine_uncertainty']

        queries = normalize_kpts(queries,'l')
        fine = normalize_kpts(fine,'r')
        queries = queries.cpu().numpy()
        fine = fine.cpu().numpy()
        unc = unc.cpu().numpy()

        fine, unc = fine[0], unc[0]
        queries = queries / scale_a[::-1] / ori_scale_a[::-1]
        fine = fine / scale_b[::-1] / ori_scale_b[::-1]
        queries = queries + np.array(q_base_coors)
        fine = fine + np.array(t_base_coors)

        res = np.concatenate([queries, fine, unc], axis=1)
        return res
    
    def forward_2stage(self, img_a, img_b, queries = None, cycle=True):
        source_img, target_img = img_a.copy(), img_b.copy()
        h0, w0, _ = source_img.shape
        h1, w1, _ = target_img.shape

        coarse_res = self.forward_coarse(source_img, target_img)
        coarse_queries, coarse_corrs = coarse_res[:, :2], coarse_res[:, 2:4]
        if len(coarse_queries) == 0:
            coarse_queries = np.stack([[w0/3, h0/3], [w0*2/3, h0*2/3]], axis=0)
            coarse_corrs = np.stack([[w1/3, h1/3], [w1*2/3, h1*2/3]], axis=0)
        q_board = get_corner(coarse_queries)
        t_board = get_corner(coarse_corrs)
        
        source_img, q_base_coors = crop_image(source_img, q_board)
        target_img, t_base_coors = crop_image(target_img, t_board)
        
        if queries is not None:
            queries = queries - np.array(q_base_coors)
        res = self.forward(source_img, target_img, queries = queries, cycle=cycle)
        res[..., :2] = res[..., :2] + np.array(q_base_coors)
        res[..., 2:4] = res[..., 2:4] + np.array(t_base_coors)
        return res
