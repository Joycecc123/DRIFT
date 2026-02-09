import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Tuple, Optional, List

INVERSEFORM_MODULE = os.path.join("checkpoints", "/home/joyce/nnUNet/nnunetv2/training/loss/__pycache__/hrnet48_OCR_IF_checkpoint.pth")

def load_model_from_dict(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    updated_model_dict = {}
    for k_model, v_model in model_dict.items():
        if k_model.startswith('model') or k_model.startswith('module'):
            k_updated = '.'.join(k_model.split('.')[1:])
            updated_model_dict[k_updated] = k_model
        else:
            updated_model_dict[k_model] = k_model

    updated_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('model') or k.startswith('modules'):
            k = '.'.join(k.split('.')[1:])
        if k in updated_model_dict.keys() and model_dict[k].shape == v.shape:
            updated_pretrained_dict[updated_model_dict[k]] = v

    model_dict.update(updated_pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def erosion2d(image, kernel):
   
    B, C, H, W = image.shape
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    patches = F.unfold(padded, kernel_size=kernel.shape)
    erosion = patches.min(dim=1)[0]
    return erosion.view(B, C, H, W)

class InverseTransform2D(nn.Module):
    def __init__(self, 
                 model_output=None, 
                 base_size=224, 
                 min_tile_size=128,
                 max_tile_size=512,
                 overlap=32,
                 direct_process_threshold=256):
        super(InverseTransform2D, self).__init__()
        
        self.base_size = base_size
        self.min_tile_size = min_tile_size
        self.max_tile_size = max_tile_size
        self.overlap = overlap
        self.direct_process_threshold = direct_process_threshold
        
       
        inversenet_backbone = InverseNet()
        self.inversenet = load_model_from_dict(inversenet_backbone, INVERSEFORM_MODULE).cuda()
        for param in self.inversenet.parameters():
            param.requires_grad = False
            
        
        self.skel_weight = 0.2
        self.dt_weight = 0.2
        self.global_weight = 0.3  
        self.inverse_weight = 0.3
        self.kernel = torch.ones(3, 3).cuda()

    def compute_skeleton_loss(self, inputs, targets):
        
        kernel = torch.ones(3, 3, device=inputs.device)
        inputs_skel = inputs - erosion2d(inputs, kernel)
        targets_skel = targets - erosion2d(targets, kernel)
        
        # 引入长程像素交互
        long_range_map = self.compute_long_range_interaction(inputs_skel, targets_skel)
        return F.mse_loss(inputs_skel, targets_skel) + self.global_weight * F.mse_loss(long_range_map, targets_skel)
        
    def compute_adaptive_distance_field_loss(self, inputs, targets, alpha=0.5, beta=0.1, max_dist=10):
        
        dist_inputs = torch.zeros_like(inputs)
        dist_targets = torch.zeros_like(targets)

        
        sigma_map = torch.ones_like(inputs) * beta  
        alpha_map = 1 / (1 + torch.exp(-alpha * targets)) 
        
        for i in range(1, max_dist):
            kernel_size = 2 * i + 1
            padding = i
            inputs_dilated = F.max_pool2d(inputs, kernel_size=kernel_size, stride=1, padding=padding)
            targets_dilated = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=padding)
            
            dist_inputs = torch.where(inputs_dilated > 0,
                                      torch.ones_like(inputs) * (1 - i / max_dist),
                                      dist_inputs)
            dist_targets = torch.where(targets_dilated > 0,
                                       torch.ones_like(targets) * (1 - i / max_dist),
                                       dist_targets)

        
        adaptive_influence_inputs = alpha_map * torch.exp(-dist_inputs / sigma_map)
        adaptive_influence_targets = alpha_map * torch.exp(-dist_targets / sigma_map)
        
      
        long_range_map = self.compute_long_range_interaction(adaptive_influence_inputs, adaptive_influence_targets)
        return F.mse_loss(adaptive_influence_inputs, adaptive_influence_targets) + self.global_weight * F.mse_loss(long_range_map, adaptive_influence_targets)

    def compute_inverse_loss(self, inputs, targets):
       
        _, _, distance_coeffs = self.inversenet(inputs, targets)
        base_loss = (((distance_coeffs * distance_coeffs).sum(dim=1)) ** 0.5).mean()
        
       
        long_range_map = self.compute_long_range_interaction(inputs, targets)
        return base_loss + self.global_weight * F.mse_loss(long_range_map, targets)

    def compute_long_range_interaction(self, inputs, bif_map):
       
        B, C, H, W = inputs.shape
        long_range_map = torch.zeros_like(bif_map)

    
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    
                    weights = torch.exp(-torch.sum((inputs[b, :, :, :] - inputs[b, :, h, w]) ** 2, dim=(1, 2)))
                    weights /= torch.sum(weights) 
                    
                    
                    long_range_map[b, :, h, w] = torch.sum(weights * bif_map[b, :, :, :])
        
        return long_range_map

    def process_single_scale(self, inputs, targets, scale_size=None):
        
        inputs = torch.clamp(inputs, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        scale_size = inputs.shape[2:]

        if scale_size[0] <= self.direct_process_threshold or scale_size[1] <= self.direct_process_threshold:
            inputs = F.interpolate(inputs, size=(self.base_size, self.base_size), mode='nearest')
            targets = F.interpolate(targets, size=(self.base_size, self.base_size), mode='nearest')
            
          
            inverse_loss = self.compute_inverse_loss(inputs, targets)
            
         
            skel_loss = self.compute_skeleton_loss(inputs, targets)
            dt_loss = self.compute_adaptive_distance_field_loss(inputs, targets)
            
            return self.inverse_weight * inverse_loss + self.skel_weight * skel_loss + self.dt_weight * dt_loss
        
        
        tile_h, tile_w = self.calculate_optimal_tile_size(scale_size[0], scale_size[1])
        num_tiles_h = (scale_size[0] + tile_h - 1) // tile_h
        num_tiles_w = (scale_size[1] + tile_w - 1) // tile_w
        
        total_loss = 0
        valid_tiles = 0
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                h_start = i * tile_h
                h_end = min((i + 1) * tile_h, scale_size[0])
                w_start = j * tile_w
                w_end = min((j + 1) * tile_w, scale_size[1])
                
                inputs_tile = inputs[:, :, h_start:h_end, w_start:w_end]
                targets_tile = targets[:, :, h_start:h_end, w_start:w_end]
                
                inputs_tile = F.interpolate(inputs_tile, size=(self.base_size, self.base_size), 
                                        mode='nearest')
                targets_tile = F.interpolate(targets_tile, size=(self.base_size, self.base_size),
                                             mode='nearest')
                
               
                inverse_loss = self.compute_inverse_loss(inputs_tile, targets_tile)
                
                skel_loss = self.compute_skeleton_loss(inputs_tile, targets_tile)
                dt_loss = self.compute_adaptive_distance_field_loss(inputs_tile, targets_tile)
                
                tile_loss = self.inverse_weight * inverse_loss + self.skel_weight * skel_loss + self.dt_weight * dt_loss
                total_loss += tile_loss
                valid_tiles += 1
        
        if valid_tiles == 0:
            raise RuntimeError(f"No valid tiles processed for scale size {scale_size}")
            
        return total_loss / valid_tiles

    def forward(self, inputs, targets):
       
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            targets = [targets]
        
        total_loss = 0
        valid_scales = 0
        
        for scale_idx, (scale_input, scale_target) in enumerate(zip(inputs, targets)):
           
            scale_input = torch.clamp(scale_input, 0, 1)
            scale_target = torch.clamp(scale_target, 0, 1)

            bif_loss = self.compute_adaptive_distance_field_loss(scale_input, scale_target)
            
            inverse_loss = self.compute_inverse_loss(scale_input, scale_target)
         
            skel_loss = self.compute_skeleton_loss(scale_input, scale_target)
            
         
            scale_loss = (self.inverse_weight * inverse_loss +
                          self.dt_weight * bif_loss +
                          self.skel_weight * skel_loss)
            
           
            scale_weight = 1.0 / (2 ** scale_idx)
            total_loss += scale_loss * scale_weight
            valid_scales += 1
        
        if valid_scales == 0:
            raise RuntimeError("No valid scales processed!")
        
        return total_loss / valid_scales

class InverseNet(nn.Module):
    def __init__(self):
        super(InverseNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(224*224*2, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )

    def forward(self, x1, x2):
        if x1.shape[-2:] != (224, 224):
            x1 = F.interpolate(x1, size=(224, 224), mode='bilinear', align_corners=True)
        if x2.shape[-2:] != (224, 224):
            x2 = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=True)
            
        x = torch.cat((x1.view(-1, 224*224), x2.view(-1, 224*224)), dim=1)
        return x1, x2, self.fc(x)

        
       
