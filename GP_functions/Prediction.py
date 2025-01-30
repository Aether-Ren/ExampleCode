"""
File: Prediction.py
Author: Hongjin Ren
Description: Predict the reslut from Gaussian process models

"""

#############################################################################
## Package imports
#############################################################################
import torch
import gpytorch



#############################################################################
## 
#############################################################################

def preds_for_one_model(model, likelihood, xxx):
    # Prediction of a column of the local data

    model.eval()
    likelihood.eval()
    # with torch.no_grad(),gpytorch.settings.fast_pred_var():
    preds = likelihood(model(xxx)).mean
    return preds

def full_preds(models, likelihoods, xxx):
    # Use the GP model to get a complete prediction of the output
    # input_point = input_point.unsqueeze(0)
    full_preds_point = preds_for_one_model(models[0], likelihoods[0], xxx).unsqueeze(1)
    for i in range(1, len(models)):
        preds = preds_for_one_model(models[i], likelihoods[i],xxx).unsqueeze(1)
        full_preds_point = torch.cat((full_preds_point, preds), 1)
    return full_preds_point.squeeze()

