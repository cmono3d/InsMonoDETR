import torch
import torch.nn as nn
import math

from .balancer import Balancer
from .focalloss import FocalLoss
from PIL import Image
import numpy as np
# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class DDNLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

    def build_target_depth_from_3dcenter(self, depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img, instances,mask_instances):

        B, _, H, W = depth_logits.shape
        depth_maps = torch.zeros((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype)

        instances_maps =[]
        # transform segmentation mask
        for img in instances:
            img = img.detach().cpu().numpy()
            img = Image.fromarray(img)
            img = img.resize((W, H),Image.NEAREST)
            instances_maps.append(np.array(img))
        instances_maps = torch.tensor(instances_maps).to(device=depth_logits.device)
        
        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.long()
        
        # Set all values within each box to True
        gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)  
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes2d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
            
            mask_instances_b = mask_instances[b][sorted_idx] # instances in dataloader
            mask_instances_b = [1000+ins for ins in mask_instances_b] #check logic, car only
            mask_instances_b = torch.tensor(mask_instances_b).to(dtype=depth_logits.dtype)

          # filter object in dataloader
            new_instance_map = torch.zeros_like(instances_maps[0]) 
            for i in range(len(mask_instances_b)):
                new_instance_map[instances_maps[b]==mask_instances_b[i]] = mask_instances_b[i]
            new_instance_map = new_instance_map.to(dtype=depth_logits.dtype)
            
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                new_instance_map[new_instance_map==mask_instances_b[n]] = center_depth_per_batch[n]
                depth_maps[b] = new_instance_map

        return depth_maps

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)
       
        return indices

    def forward(self, depth_logits, gt_boxes2d, num_gt_per_img, gt_center_depth,**kwargs):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """
        instances = kwargs['instances']
        mask_instances = kwargs['mask']

        # Bin depth map to create target
        depth_maps = self.build_target_depth_from_3dcenter(depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img,instances,mask_instances)
        #ipdb.set_trace()
        depth_target = self.bin_depths(depth_maps, target=True)
        #ipdb.set_trace()
        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)
        #ipdb.set_trace()
        # Compute foreground/background balancing
        loss = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)

        return loss
