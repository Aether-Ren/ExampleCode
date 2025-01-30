"""
File: Loss_function.py
Author: Hongjin Ren
Description: Create the loss function (Euclid, ...)

"""

#############################################################################
## Package imports
#############################################################################
import torch
import GP_functions.Prediction as Prediction


#############################################################################
## 
#############################################################################



def surrogate_loss_euclid(params, models, likelihoods, row_idx, test_y):
    
    with torch.no_grad():

        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available():
            params_tensor = params_tensor.cuda()

        pred = Prediction.full_preds(models, likelihoods, params_tensor)
        loss = torch.norm(pred - test_y[row_idx,:]).pow(2).item()

    return loss



def euclidean_distance_loss(output, target):
    return torch.norm(output - target).pow(2)
