"""
File: bound.py
Author: Hongjin Ren
Description: Find the bound of paramaters in the local training part

"""

#############################################################################
## Package imports
#############################################################################
import torch
from scipy.optimize import Bounds



#############################################################################
## 
#############################################################################


def get_bounds(tensor):
    col_min_values, _ = torch.min(tensor, dim=0)
    col_max_values, _ = torch.max(tensor, dim=0)
    bounds = Bounds(col_min_values.cpu().numpy(), col_max_values.cpu().numpy())
    bounds_list = list(zip(bounds.lb, bounds.ub))

    return bounds_list