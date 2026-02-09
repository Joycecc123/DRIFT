import torch
from torch import nn



class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
       
        super(DeepSupervisionWrapper, self).__init__()
        if weight_factors is not None:
            assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
            self.weight_factors = tuple(weight_factors)
        else:
            self.weight_factors = None
        self.loss = loss

    def forward(self, *args):
    
        
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"

        weights = self.weight_factors if self.weight_factors is not None else (1,) * len(args[0])
        sum_total_loss = 0
        sum_dc_loss = 0
        sum_ce_loss = 0
        sum_inverse_loss = 0

        for i, inputs in enumerate(zip(*args)):
            if weights[i] != 0.0:
                total_loss, dc_loss, ce_loss, inverse_loss = self.loss(*inputs) 
                sum_total_loss += total_loss
                sum_dc_loss += dc_loss
                sum_ce_loss += ce_loss
                sum_inverse_loss += inverse_loss

        avg_total_loss = sum_total_loss / len(weights)
        avg_dc_Loss = sum_dc_loss / len(weights)
        avg_ce_loss = sum_ce_loss / len(weights)
        avg_inverse_loss = sum_inverse_loss / len(weights)

        return avg_total_loss, avg_dc_Loss, avg_ce_loss, avg_inverse_loss
    