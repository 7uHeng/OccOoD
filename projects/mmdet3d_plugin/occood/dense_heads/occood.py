# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jianbiao Mei
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, builder
from projects.mmdet3d_plugin.occood.utils.header import Header, SparseHeader
from projects.mmdet3d_plugin.occood.modules.sgb import Sem_RWKV
from projects.mmdet3d_plugin.occood.modules.sdb import SDB
from projects.mmdet3d_plugin.occood.modules.flosp import FLoSP
from projects.mmdet3d_plugin.occood.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.occood.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss, CE_loss_2D
from projects.mmdet3d_plugin.occood.modules.occ_rwkv import Occbranch
from projects.mmdet3d_plugin.occood.modules.bev_net import BEVUNetv1
from projects.mmdet3d_plugin.occood.utils.ssc_loss_pts import sem_scal_loss_pts
from projects.mmdet3d_plugin.occood.modules.sem import Sem_Decoder
def get_sparse_labels(dense_volume, sparse_indices):
    """
    Args:
        dense_volume: shape (B, D, H, W)
        sparse_indices: shape (N, 4), columns: [batch_id, z, y, x]
    Returns:
        sparse_labels: shape (N, )
    """
    b_ids = sparse_indices[:, 0].long()
    z_ids = sparse_indices[:, 1].long()
    y_ids = sparse_indices[:, 2].long()
    x_ids = sparse_indices[:, 3].long()
    return dense_volume[b_ids, z_ids, y_ids, x_ids]
class AttentiveFusion(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(n_classes * 2, n_classes // 4, kernel_size=1),
            nn.BatchNorm3d(n_classes // 4),
            nn.ReLU(),
            nn.Conv3d(n_classes // 4, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, bev_logit, vox_logit):
        gate_weights = self.gate(torch.cat([bev_logit, vox_logit], dim=1))
        return vox_logit + gate_weights * bev_logit
@HEADS.register_module()
class OccOoDHead(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
        scale_2d_list,
        pts_header_dict,
        depth=3,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = False,
        ood_flag = False,
        save_flag_ood = False,
        use_sem = True,
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.embed_dims = embed_dims

        if kwargs.get('dataset', 'semantickitti') == 'semantickitti':#1，2，3，4，5，6，7，8
            self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                                "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
            self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        elif kwargs.get('dataset', 'semantickitti') == 'kitti360':
            self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object']
            self.class_weights = torch.from_numpy(np.array([0.464, 0.595, 0.865, 0.871, 0.717, 0.657, 0.852, 0.541, 0.602, 0.567, 0.607, 0.540, 0.636, 0.513, 0.564, 0.701, 0.774, 0.580, 0.690]))
        self.n_classes = len(self.class_names)

        self.flosp = FLoSP(scale_2d_list)
        self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)
        self.sem_rwkv_branch = Sem_RWKV(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims)
        self.mlp_prior = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.LayerNorm(self.embed_dims//2),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dims//2, self.embed_dims)
        )
        occ_channel = 8 if pts_header_dict.get('guidance', False) else 0
        self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=self.embed_dims//2, depth=depth)
        self.occ_rwkv_branch = Occbranch(self.embed_dims, bev_h, bev_w, bev_z)
        self.occ_header = nn.Sequential(
            SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=1),
            nn.Conv3d(self.embed_dims//2, 1, kernel_size=3, padding=1)
        )
        self.sem_header = SparseHeader(self.n_classes, feature=self.embed_dims)
        self.ssc_header = Header(self.n_classes, feature=self.embed_dims//2)
        self.dilation = 1
        self.bilinear = True
        self.group_conv = False
        self.input_batch_norm = True
        self.dropout = 0.5
        self.circular_padding = False
        self.dropblock = False
        self.pts_header = builder.build_head(pts_header_dict)
        self.bevbranch = BEVUNetv1(self.n_classes*self.bev_z, self.bev_z, self.dilation, self.bilinear, self.group_conv,
                            self.input_batch_norm, self.dropout, self.circular_padding, self.dropblock)
        self.reduction = nn.Sequential(
            nn.Conv2d(128 * 16, 128, kernel_size=1, groups=16),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fusion_attn = AttentiveFusion(n_classes = self.n_classes)
        if self.use_sem:
            self.sem_decoder = Sem_Decoder(input_channel=self.embed_dims, num_classes=self.n_classes, ratio=2)
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
        self.save_flag_ood =save_flag_ood
        self.ood_flag = ood_flag
        self.use_sem = use_sem

    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """
        out = {}

        if self.use_sem:
            sem_feats = mlvl_feats[-1].squeeze(1)
            if len(sem_feats.shape) == 5:
                sem_feats = sem_feats[:, -1, :, :, :]
            sem_out = self.sem_decoder(sem_feats)
            out["sem"] = sem_out.unsqueeze(0)


        x3d = self.flosp(mlvl_feats, img_metas) # bs, c, nq
        bs, c, _ = x3d.shape
        x3d = self.bottleneck(x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z))
        occ = self.occ_header(x3d).squeeze(1)
        out["occ"] = occ 

        # geometry paring
        geo_bev_dense = self.occ_rwkv_branch(x3d)
        out['geo_occ_2'], out['geo_occ_4'], out['geo_occ_8'] = geo_bev_dense['mss_logits_list']

        x3d = x3d.reshape(bs, c, -1)
        # Load proposals
        pts_out = self.pts_header(mlvl_feats, img_metas, target)
        pts_occ = pts_out['occ_logit'].squeeze(1)
        proposal =  (pts_occ > 0).float().detach().cpu().numpy()
        out['pts_occ'] = pts_occ

        occ_full = pts_out['occ_full'].squeeze(1)
        out['pts_occ_full'] = occ_full


        if proposal.sum() < 2:
            proposal = np.ones_like(proposal)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1)>0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1)==0)).astype(np.int32)
        vox_coords = self.get_voxel_indices()

        # Compute seed features
        seed_feats = x3d[0, :, vox_coords[unmasked_idx[0], 3]].permute(1, 0)
        seed_coords = vox_coords[unmasked_idx[0], :3]
        coords_torch = torch.from_numpy(np.concatenate(
            [np.zeros_like(seed_coords[:, :1]), seed_coords], axis=1)).to(seed_feats.device)
        
        # semantic_refinemant
        seed_feats_desc, sem_bev_dense = self.sem_rwkv_branch(seed_feats, coords_torch)
        out['sem_2'], out['sem_4'], out['sem_8'] = sem_bev_dense['mss_logits_list']
        out['coord_2'], out['coord_4'], out['coord_8'] = sem_bev_dense['coord_list']
        sem = self.sem_header(seed_feats_desc)
        out["sem_logit"] = sem
        out["coords"] = seed_coords

        # cross-view_feature_synergy
        x3d_bev = x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z)
        x3d_bev = x3d_bev.permute(0, 1, 4, 2, 3) # [B, 128, 16, 128, 128]
        bev_x3d = self.reduction(x3d_bev.flatten(1, 2)) # B, 128, 128, 128
        x = self.bevbranch(bev_x3d, sem_bev_dense['mss_bev_dense'], geo_bev_dense['mss_bev_dense']) # bev3d logits

        # Complete voxel features
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=x3d.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats_desc
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mlp_prior(x3d[0, :, vox_coords[masked_idx[0], 3]].permute(1, 0))

        vox_feats_diff = vox_feats_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)
        if self.pts_header.guidance:
            vox_feats_diff = torch.cat([vox_feats_diff, pts_out['occ_x']], dim=1)
        vox_feats_diff = self.sdb(vox_feats_diff) # 1, C, H, W, Z
        ssc_dict = self.ssc_header(vox_feats_diff)

        ssc_dict["ssc_f_logit"] = self.fusion_attn(x, ssc_dict["ssc_logit"])
        
        out.update(ssc_dict)
        
        return out

    def step(self, out_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """

        ssc_pred = out_dict["ssc_f_logit"]
        ssc_pred_bev = out_dict["bev_logit"]
        ssc_pred_vox = out_dict["ssc_logit"]
        if step_type== "train":
            sem_pred_2 = out_dict["sem_logit"]

            target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
            coords = out_dict['coords']
            sp_target_2 = target_2.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2]]
            loss_dict = dict()

            class_weight = self.class_weights.type_as(target)
            if self.CE_ssc_loss:
                loss_ssc_bev = CE_ssc_loss(ssc_pred_bev, target, class_weight)
                loss_ssc_vox = CE_ssc_loss(out_dict["ssc_logit"], target, class_weight)
                loss_ssc_fus = CE_ssc_loss(out_dict["ssc_f_logit"], target, class_weight)

                loss_dict['loss_ssc_bev'] = loss_ssc_bev * 0.3
                loss_dict['loss_ssc_vox'] = loss_ssc_vox * 0.7
                loss_dict['loss_ssc'] = loss_ssc_fus * 1.0
 
            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=1), sp_target_2, ignore=255)
            loss_sem += F.cross_entropy(sem_pred_2, sp_target_2.long(), ignore_index=255)
            loss_dict['loss_sem'] = loss_sem

            ones = torch.ones_like(target_2).to(target_2.device)
            target_2_binary = torch.where(torch.logical_or(target_2==255, target_2==0), target_2, ones)
            loss_occ = F.binary_cross_entropy(out_dict['occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())
            loss_dict['loss_occ'] = loss_occ

            loss_dict['loss_pts'] = F.binary_cross_entropy(out_dict['pts_occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())

            ones_1 = torch.ones_like(target).to(target.device)
            target_binary = torch.where(torch.logical_or(target==255, target==0), target, ones_1)
            loss_dict['loss_pts_full'] = F.binary_cross_entropy(out_dict['pts_occ_full'].sigmoid()[target_binary!=255], target_binary[target_binary!=255].float())

            target_8 = torch.from_numpy(img_metas[0]['target_1_8']).unsqueeze(0).to(target.device)

            ones_2 = torch.ones_like(target_2).to(target_2.device)
            target_2_binary = torch.where(torch.logical_or(target_2==255, target_2==0), target_2, ones_2)
            ones_8 = torch.ones_like(target_8).to(target_8.device)
            target_8_binary = torch.where(torch.logical_or(target_8==255, target_8==0), target_8, ones_8)

            loss_occ_2 = F.binary_cross_entropy(out_dict['geo_occ_2'].squeeze(1).sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())
            loss_dict['loss_occ_2'] = loss_occ_2 * 0.3
            loss_occ_8 = F.binary_cross_entropy(out_dict['geo_occ_8'].squeeze(1).sigmoid()[target_8_binary!=255], target_8_binary[target_8_binary!=255].float())
            loss_dict['loss_occ_8'] = loss_occ_8 * 0.1

            vw1_coord = out_dict['coord_2']
            vw3_coord = out_dict['coord_8']

            vw_label_02 = get_sparse_labels(target_2.permute(0, 3, 1, 2), vw1_coord)
            vw_label_08 = get_sparse_labels(target_8.permute(0, 3, 1, 2), vw3_coord)  

            sem_pred_2 = out_dict['sem_2']
            sem_pred_8 = out_dict['sem_8']

            sem_scal_loss_val_2 = sem_scal_loss_pts(sem_pred_2, vw_label_02)  # Calculate semantic scale loss
            sem_scal_loss_val_8 = sem_scal_loss_pts(sem_pred_8, vw_label_08)

            loss_dict['loss_sem_2'] = sem_scal_loss_val_2 * 0.3
            loss_dict['loss_sem_8'] = sem_scal_loss_val_8 * 0.1


            if self.use_sem:
                sem_pred = out_dict["sem"]
                target_2d = torch.from_numpy(img_metas[0]["semantic"]).unsqueeze(0).to(ssc_pred.device)
                ce_loss = CE_loss_2D(sem_pred, target_2d, 4)
                loss_dict['loss_2d_ce'] = ce_loss

            return loss_dict

        elif step_type== "val" or "test":
            result = dict()
            result['output_voxels'] = ssc_pred
            result['target_voxels'] = target
            if self.ood_flag :
                # geometry prior
                y_pred = ssc_pred.detach().cpu().numpy()
                predicted_classes = np.argmax(y_pred, axis=1)
                # empty
                empty_mask = (predicted_classes == 0) # (batch_size, depth, height, width)\
                # semkitti
                # class
                class_masks = {
                    1: (predicted_classes == 1),  # "car"
                    2: (predicted_classes == 2),  # "bicycle"
                    3: (predicted_classes == 3),  # "motorcycle"
                    4: (predicted_classes == 4),  # "truck"
                    5: (predicted_classes == 5),  # "other-vehicle"
                    6: (predicted_classes == 6),  # "person"
                    7: (predicted_classes == 7),  # "bicyclist"
                    8: (predicted_classes == 8),  # "motorcyclist"
                    9: (predicted_classes == 9),  # "road"
                    10: (predicted_classes == 10),  # "parking"
                    11: (predicted_classes == 11),  # "sidewalk"
                    12: (predicted_classes == 12),  # "other-ground"
                    13: (predicted_classes == 13),  # "building"
                    14: (predicted_classes == 14),  # "fence"
                    15: (predicted_classes == 15),  # "vegetation"
                    16: (predicted_classes == 16),  # "trunk"
                    17: (predicted_classes == 17),  # "terrain"
                    18: (predicted_classes == 18),  # "pole"
                    19: (predicted_classes == 19),  # "traffic-sign"
                }
                
                ood_pred_all = torch.zeros_like(ssc_pred[:, 0, :, :, :])  # (batch_size, depth, height, width)
                instance_classes = [1, 2, 3, 4, 5, 6, 7, 8]  # instance
                region_classes = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # region
            
                for class_idx, mask in class_masks.items():
                    mask = torch.tensor(mask, device=ssc_pred.device) # (batch_size, depth, height, width)
                    mask = mask.unsqueeze(1) # (batch_size, 1, depth, height, width)
                    class_voxels = (ssc_pred * mask).squeeze(0) # (20, depth, height, width)
                    if class_idx in instance_classes:
                        # instance：cosine_similarity
                        class_ood_pred = get_cosine_similarity(class_voxels).unsqueeze(0) # (bs, depth, height, width)
                        min_val = class_ood_pred.min()
                        max_val = class_ood_pred.max()
                        class_ood_pred = (class_ood_pred - min_val) / (max_val - min_val)
                    elif class_idx in region_classes:
                        # region：entropy
                        class_voxels = F.softmax(class_voxels, dim=0) # (20, depth, height, width)
                        class_ood_pred = get_entropy(class_voxels).unsqueeze(0)  # (bs, depth, height, width)
                        min_val = class_ood_pred.min()
                        max_val = class_ood_pred.max()
                        class_ood_pred = (class_ood_pred - min_val) / (max_val - min_val)
                        class_ood_pred = class_ood_pred/2
                    mask = mask.squeeze(1) # (batch_size, 1, depth, height, width)
                    ood_pred_all[mask] = class_ood_pred[mask]

                # # sscbench-kitti-360
                # class_masks = {
                #     1: (predicted_classes == 1),  # "car"
                #     2: (predicted_classes == 2),  # "bicycle"
                #     3: (predicted_classes == 3),  # "motorcycle"
                #     4: (predicted_classes == 4),  # "truck"
                #     5: (predicted_classes == 5),  # "other-vehicle"
                #     6: (predicted_classes == 6),  # "person"
                #     7: (predicted_classes == 7),  # "road"
                #     8: (predicted_classes == 8),  # "parking"
                #     9: (predicted_classes == 9),  # "sidewalk"
                #     10: (predicted_classes == 10),  # "other-ground"
                #     11: (predicted_classes == 11),  # "building"
                #     12: (predicted_classes == 12),  # "fence"
                #     13: (predicted_classes == 13),  # "vegetation"
                #     14: (predicted_classes == 14),  # "terrain"
                #     15: (predicted_classes == 15),  # "pole"
                #     16: (predicted_classes == 16),  # "traffic-sign"
                #     17: (predicted_classes == 17),  # "other-structure"
                #     18: (predicted_classes == 18),  # "other-object"
                # }

                # ood_pred_all = torch.zeros_like(ssc_pred[:, 0, :, :, :])  # (batch_size, depth, height, width)

                # instance_classes = [1, 2, 3, 4, 5, 6]  # instance
                # region_classes = [7,8,9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # region

                # for class_idx, mask in class_masks.items():
                #     mask = torch.tensor(mask, device=ssc_pred.device)  # (batch_size, depth, height, width)
                #     mask = mask.unsqueeze(1)  # (batch_size, 1, depth, height, width)
                #     class_voxels = (ssc_pred * mask).squeeze(0)  # (20, depth, height, width)
                #     if class_idx in instance_classes:
                #         class_ood_pred = get_cosine_similarity(class_voxels).unsqueeze(0)  # (bs, depth, height, width)
                #         min_val = class_ood_pred.min()
                #         max_val = class_ood_pred.max()
                #         class_ood_pred = (class_ood_pred - min_val) / (max_val - min_val)
                #     elif class_idx in region_classes:
                #         class_voxels = F.softmax(class_voxels, dim=0)  # softmax (20, depth, height, width)
                #         class_ood_pred = get_entropy(class_voxels).unsqueeze(0)  # (bs, depth, height, width)
                #         min_val = class_ood_pred.min()
                #         max_val = class_ood_pred.max()
                #         class_ood_pred = (class_ood_pred - min_val) / (max_val - min_val)
                #         class_ood_pred = class_ood_pred / 2
                #     mask = mask.squeeze(1)  # (batch_size, depth, height, width)
                #     ood_pred_all[mask] = class_ood_pred[mask]
            
                min_ood_score = ood_pred_all.min()
                empty_mask_expanded = torch.tensor(empty_mask,device=ssc_pred.device) # (1, depth, height, width)
                ood_pred_all[empty_mask_expanded] = min_ood_score # empty min_ood_score
                ood_pred = ood_pred_all
                result['ood_pred'] = ood_pred            

                if self.save_flag_ood:
                    min_val = ood_pred.min()
                    max_val = ood_pred.max()
                    ood_pred = (ood_pred - min_val) / (max_val - min_val)
                    ood_scores_mapped = torch.tensor((ood_pred.cpu().numpy() * 255).astype(np.uint8))
                    zero_mask_mapped = ood_scores_mapped == 0
                    ood_scores_mapped[zero_mask_mapped] += 1
                    ood_scores_mapped[empty_mask_expanded] =0
                    ood_scores_tensor = torch.tensor(ood_scores_mapped, dtype=torch.int64).unsqueeze(0)
                    y_pred = ood_scores_tensor.detach().cpu().numpy()
                    self.save_ood_pred(img_metas, y_pred)       

            if self.save_flag:
                y_pred = ssc_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                self.save_pred(img_metas, y_pred)

            return result

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, target, img_metas, "val")

    def get_voxel_indices(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        return vox_coords
    def save_ood_pred(self, img_metas, y_pred):
        # save predictions
        pred_folder = os.path.join("/root/autodl-tmp/mmdetection3d/prediction_ood_stu", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))    
    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        20: 100 # "anomaly"
        Note: only for semantickitti
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40
        # y_pred[y_pred==20] = 100

        # save predictions
        pred_folder = os.path.join("/root/autodl-tmp/prediction", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))

def get_rba(logit):
    return -logit.tanh().sum(dim=0) # [w, l, h]

def get_energy(logit):
    return -torch.logsumexp(logit, dim=0)

def get_postpro(logit):
    return 1-torch.max(logit,dim=0)[0]

def get_entropy(logit):
    entropy = -torch.sum(logit * torch.log(logit + 1e-8), dim=0)
    return entropy

def get_min_probability(logit):
    softmax_prob = torch.softmax(logit, dim=0)
    min_prob = torch.min(softmax_prob, dim=0)[0]
    return 1 - min_prob

def get_topk_probability(logit, k=3):
    softmax_prob = torch.softmax(logit, dim=0)
    topk_probs = torch.topk(softmax_prob, k=k, dim=0).values
    return 1 - torch.sum(topk_probs, dim=0)

def get_kl_divergence(logit):
    uniform_dist = torch.ones_like(logit) / logit.size(0)
    kl_div = torch.sum(logit * torch.log(logit / uniform_dist + 1e-8), dim=0)
    return kl_div

def get_cosine_similarity(logit):
    mean_logit = torch.mean(logit, dim=0, keepdim=True)
    cosine_sim = torch.nn.functional.cosine_similarity(logit, mean_logit, dim=0)
    return 1 - cosine_sim

def get_spatial_consistency(logit, kernel_size=3):
    # logit: [20, 256, 256, 32]
    unfolded = torch.nn.functional.unfold(logit, kernel_size, padding=kernel_size//2)
    variance = unfolded.var(dim=0)  
    return variance 