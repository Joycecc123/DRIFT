import torch
import numpy as np
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.InverseForm import InverseTransform2D
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from typing import Optional, Dict, Tuple, List, Union
import torch.nn.functional as F


class DC_and_CE_loss_backup(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
    
##——————————————————————————————————————————————————————————————————————————##
##———————————————————————————————带有逆变损失的—————————————————————————————————##
class DC_and_CE_loss_1(nn.Module):
    def __init__(self, 
                 soft_dice_kwargs: Dict, 
                 ce_kwargs: Dict,
                 inverse_kwargs: Optional[Dict] = None,
                 weight_ce: float = 1.0,
                 weight_dice: float = 1.0,
                 weight_inverse: float = 1.0,
                 ignore_label: Optional[int] = None,
                 dice_class: nn.Module = None):
        super(DC_and_CE_loss, self).__init__()
        
        ce_kwargs_clean = {}
        if ce_kwargs is not None:
            ce_kwargs_clean = ce_kwargs.copy()
            for param in ['batch_dice', 'do_bg', 'ddp', 'smooth']:
                ce_kwargs_clean.pop(param, None)
        
        if ignore_label is not None:
            ce_kwargs_clean['ignore_index'] = ignore_label

        self.weight_dice = float(weight_dice)
        self.weight_ce = float(weight_ce)
        self.weight_inverse = float(weight_inverse)
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs_clean)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        
        if inverse_kwargs is not None:
            self.inverse_loss = InverseTransform2D(**inverse_kwargs)
        else:
            self.inverse_loss = None

    def compute_loss_for_scale(self, 
                          net_output: torch.Tensor, 
                          target: torch.Tensor,
                          edge_output_class1: Optional[torch.Tensor] = None,
                          edge_output_class2: Optional[torch.Tensor] = None,
                          edge_gt_class1: Optional[torch.Tensor] = None,
                          edge_gt_class2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算单个尺度的损失"""
        # 计算 DC loss
        if self.weight_dice != 0:
            if self.ignore_label is not None:
                mask = target != self.ignore_label
                target_dice = torch.where(mask, target, torch.zeros_like(target))
            else:
                target_dice = target
                mask = None
            dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        else:
            dc_loss = torch.tensor(0.0, device=target.device)

        if self.weight_ce != 0:
            ce_loss = self.ce(net_output, target[:, 0].long())
        else:
            ce_loss = torch.tensor(0.0, device=target.device)

        # 计算 Inverse loss
        inverse_loss = torch.tensor(0.0, device=target.device)
        if self.inverse_loss is not None and edge_output_class1 is not None:
            # 打印原始edge输出和gt的信息
            # print(f"\nEdge outputs original stats:")
            # print(f"Edge output 1 range: [{edge_output_class1.min():.4f}, {edge_output_class1.max():.4f}]")
            # print(f"Edge output 2 range: [{edge_output_class2.min():.4f}, {edge_output_class2.max():.4f}]")
            # print(f"Edge GT 1 range: [{edge_gt_class1.min():.4f}, {edge_gt_class1.max():.4f}]")
            # print(f"Edge GT 2 range: [{edge_gt_class2.min():.4f}, {edge_gt_class2.max():.4f}]")

            # 确保输入在合理范围内
            edge_output_class1 = torch.clamp(edge_output_class1, 0, 1)
            edge_output_class2 = torch.clamp(edge_output_class2, 0, 1)
            
            # 添加一个很小的值以避免数值问题
            epsilon = 1e-7
            edge_output_class1 = edge_output_class1 + epsilon
            edge_output_class2 = edge_output_class2 + epsilon
            
            # 再次打印处理后的值
            # print(f"\nEdge outputs after processing:")
            # print(f"Edge output 1 range: [{edge_output_class1.min():.4f}, {edge_output_class1.max():.4f}]")
            # print(f"Edge output 2 range: [{edge_output_class2.min():.4f}, {edge_output_class2.max():.4f}]")
            
            # 计算每个类别的损失
            inverse_loss_1 = self.inverse_loss(edge_output_class1, edge_gt_class1)
            inverse_loss_2 = self.inverse_loss(edge_output_class2, edge_gt_class2)
            inverse_loss = 0.5 * (inverse_loss_1 + inverse_loss_2)
            
            # print(f"Inverse loss 1: {inverse_loss_1.item():.6f}")
            # print(f"Inverse loss 2: {inverse_loss_2.item():.6f}")

        return dc_loss, ce_loss, inverse_loss

    def forward(self, 
                net_output: List[torch.Tensor],
                target: List[torch.Tensor],
                edge_output_class1: Optional[List[torch.Tensor]] = None,
                edge_output_class2: Optional[List[torch.Tensor]] = None,
                edge_gt_class1: Optional[List[torch.Tensor]] = None,
                edge_gt_class2: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算多尺度损失
        
        Args:
            net_output: 列表，包含不同尺度的网络输出 [N, C, H, W]
            target: 列表，包含不同尺度的目标 [N, 1, H, W]
            edge_output_class1: 可选，列表，包含不同尺度的边缘输出1
            edge_output_class2: 可选，列表，包含不同尺度的边缘输出2
            edge_gt_class1: 可选，列表，包含不同尺度的边缘真值1
            edge_gt_class2: 可选，列表，包含不同尺度的边缘真值2
        """
        device = target[0].device
        total_dc_loss = torch.tensor(0.0, device=device)
        total_ce_loss = torch.tensor(0.0, device=device)
        total_inverse_loss = torch.tensor(0.0, device=device)
        
        # 确保所有输入长度一致
        assert len(net_output) == len(target), "net_output and target must have the same length"
        if edge_output_class1 is not None:
            assert len(edge_output_class1) == len(target), "edge_output_class1 must have the same length as target"
            assert len(edge_output_class2) == len(target), "edge_output_class2 must have the same length as target"
            assert len(edge_gt_class1) == len(target), "edge_gt_class1 must have the same length as target"
            assert len(edge_gt_class2) == len(target), "edge_gt_class2 must have the same length as target"
    
        dc_loss, ce_loss, inverse_loss = self.compute_loss_for_scale(
            net_output,
            target,
            edge_output_class1,
            edge_output_class2,
            edge_gt_class1,
            edge_gt_class2,
        )

        total_dc_loss += dc_loss
        total_ce_loss += ce_loss
        total_inverse_loss += inverse_loss

        # 计算平均损失
        total_dc_loss = total_dc_loss
        total_ce_loss = total_ce_loss
        total_inverse_loss = total_inverse_loss

        # 计算加权总损失
        total_loss = (self.weight_ce * total_ce_loss + 
                     self.weight_dice * total_dc_loss + 
                     self.weight_inverse * total_inverse_loss)

        return total_loss, total_dc_loss, total_ce_loss, total_inverse_loss
    
class DC_and_CE_loss(nn.Module):
    def __init__(self, 
                 soft_dice_kwargs: Dict, 
                 ce_kwargs: Dict,
                 inverse_kwargs: Optional[Dict] = None,
                 weight_ce: float = 1.0,
                 weight_dice: float = 1.0,
                 weight_inverse: float = 1.0,
                 ignore_label: Optional[int] = None,
                 dice_class: nn.Module = None):
        super(DC_and_CE_loss, self).__init__()
       
    
        ce_kwargs_clean = {}
        if ce_kwargs is not None:
            ce_kwargs_clean = ce_kwargs.copy()
            for param in ['batch_dice', 'do_bg', 'ddp', 'smooth']:
                ce_kwargs_clean.pop(param, None)

        if ignore_label is not None:
            ce_kwargs_clean['ignore_index'] = ignore_label

        self.weight_dice = float(weight_dice)
        self.weight_ce = float(weight_ce)
        self.weight_inverse = float(weight_inverse)
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs_clean)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        
        if inverse_kwargs is not None:
            self.inverse_loss = InverseTransform2D(**inverse_kwargs)
        else:
            self.inverse_loss = None

        # 增加一个正则化项权重
        self.regularization_weight = 0.001
        
    

    def compute_loss_for_scale(self, 
                               net_output: torch.Tensor, 
                               target: torch.Tensor,
                               edge_output_class1: Optional[torch.Tensor] = None,
                               edge_output_class2: Optional[torch.Tensor] = None,
                               edge_gt_class1: Optional[torch.Tensor] = None,
                               edge_gt_class2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算单个尺度的损失"""

        # 计算 DC loss
        if self.weight_dice != 0:
            if self.ignore_label is not None:
                mask = target != self.ignore_label
                target_dice = torch.where(mask, target, torch.zeros_like(target))
            else:
                target_dice = target
                mask = None
            dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        else:
            dc_loss = torch.tensor(0.0, device=target.device)

        # 计算 Cross Entropy loss
        if self.weight_ce != 0:
            ce_loss = self.ce(net_output, target[:, 0].long())
        else:
            ce_loss = torch.tensor(0.0, device=target.device)

        # 计算双向 Inverse loss
        inverse_loss = torch.tensor(0.0, device=target.device)
        if self.inverse_loss is not None and edge_output_class1 is not None:
            # 添加平滑项以正则化边缘输出
            epsilon = 1e-7

            # 确保输入在合理范围内
            edge_output_class1 = torch.clamp(edge_output_class1, 0, 1)
            edge_output_class2 = torch.clamp(edge_output_class2, 0, 1)
            
            # 添加一个很小的值以避免数值问题
            edge_output_class1 = edge_output_class1 + epsilon
            edge_output_class2 = edge_output_class2 + epsilon

            # 只对边缘区域进行计算，以减少对背景的影响
            edge_mask = (edge_gt_class1 > 0) | (edge_gt_class2 > 0)
            edge_output_class1 = edge_output_class1 * edge_mask.float()
            edge_output_class2 = edge_output_class2 * edge_mask.float()
            
            # 计算每个类别的损失：从输出边缘到真值边缘
            inverse_loss_1 = self.inverse_loss(edge_output_class1, edge_gt_class1)
            inverse_loss_2 = self.inverse_loss(edge_output_class2, edge_gt_class2)
            
            # 增加计算：从真值边缘到输出边缘的损失
            inverse_loss_gt_to_output_1 = self.inverse_loss(edge_gt_class1, edge_output_class1)
            inverse_loss_gt_to_output_2 = self.inverse_loss(edge_gt_class2, edge_output_class2)
            
            # 结合两个方向的损失，取平均值
            inverse_loss = 0.25 * (inverse_loss_1 + inverse_loss_2 + inverse_loss_gt_to_output_1 + inverse_loss_gt_to_output_2)

        return dc_loss, ce_loss, inverse_loss

    def forward(self, 
                net_output: List[torch.Tensor],
                target: List[torch.Tensor],
                edge_output_class1: Optional[List[torch.Tensor]] = None,
                edge_output_class2: Optional[List[torch.Tensor]] = None,
                edge_gt_class1: Optional[List[torch.Tensor]] = None,
                edge_gt_class2: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算多尺度损失

        Args:
            net_output: 列表，包含不同尺度的网络输出 [N, C, H, W]
            target: 列表，包含不同尺度的目标 [N, 1, H, W]
            edge_output_class1: 可选，列表，包含不同尺度的边缘输出1
            edge_output_class2: 可选，列表，包含不同尺度的边缘输出2
            edge_gt_class1: 可选，列表，包含不同尺度的边缘真值1
            edge_gt_class2: 可选，列表，包含不同尺度的边缘真值2
            current_epoch: 可选，当前训练的 epoch 数，用于动态调整权重
        """
        device = target[0].device
        total_dc_loss = torch.tensor(0.0, device=device)
        total_ce_loss = torch.tensor(0.0, device=device)
        total_inverse_loss = torch.tensor(0.0, device=device)

       

        # 确保所有输入长度一致
        assert len(net_output) == len(target), "net_output and target must have the same length"
        if edge_output_class1 is not None:
            assert len(edge_output_class1) == len(target), "edge_output_class1 must have the same length as target"
            assert len(edge_output_class2) == len(target), "edge_output_class2 must have the same length as target"
            assert len(edge_gt_class1) == len(target), "edge_gt_class1 must have the same length as target"
            assert len(edge_gt_class2) == len(target), "edge_gt_class2 must have the same length as target"

        # 计算每个尺度的损失并累加
        for i in range(len(target)):
            dc_loss, ce_loss, inverse_loss = self.compute_loss_for_scale(
                net_output[i],
                target[i],
                edge_output_class1[i] if edge_output_class1 is not None else None,
                edge_output_class2[i] if edge_output_class2 is not None else None,
                edge_gt_class1[i] if edge_gt_class1 is not None else None,
                edge_gt_class2[i] if edge_gt_class2 is not None else None,
            )

            total_dc_loss += dc_loss
            total_ce_loss += ce_loss
            total_inverse_loss += inverse_loss

        # 计算平均损失
        total_dc_loss = total_dc_loss / len(target)
        total_ce_loss = total_ce_loss / len(target)
        total_inverse_loss = total_inverse_loss / len(target)

        # 计算加权总损失
        total_loss = (self.weight_ce * total_ce_loss + 
                     self.weight_dice * total_dc_loss + 
                     self.weight_inverse * total_inverse_loss)

        return total_loss, total_dc_loss, total_ce_loss, total_inverse_loss


    
class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
