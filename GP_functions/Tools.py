"""
File: Tools.py
Author: Hongjin Ren
Description: Some tools which can help analyze some data, I wish.

"""

#############################################################################
## Package imports
#############################################################################

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.spatial import distance
from scipy.stats import qmc, multivariate_normal
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.sparse import random as sparse_random


#############################################################################
## 
#############################################################################


def Print_percentiles(mse_array):
    """
    Prints the 1st, 2nd, and 3rd quantiles of the given data.
    """
    return {
        '25th Perc.': np.percentile(mse_array, 25),
        'Median': np.percentile(mse_array, 50),
        '75th Perc.': np.percentile(mse_array, 75)
    }



#############################################################################
## 
#############################################################################

def select_subsequence(original_points, target_num_points):

    # Calculate the step to select points to approximately get the target number of points
    total_points = len(original_points)
    step = max(1, total_points // target_num_points)
    
    # Select points by stepping through the original sequence
    selected_points = original_points[::step]
    
    # Ensure we have exactly target_num_points by adjusting the selection if necessary
    if len(selected_points) > target_num_points:
        # If we selected too many points, trim the excess
        selected_points = selected_points[:target_num_points]
    elif len(selected_points) < target_num_points:
        # If we selected too few points, this indicates a rounding issue with step; handle as needed
        # This is a simple handling method and might need refinement based on specific requirements
        additional_indices = np.random.choice(range(total_points), size=target_num_points - len(selected_points), replace=False)
        additional_points = original_points[additional_indices]
        selected_points = np.vstack((selected_points, additional_points))
    
    return selected_points 






#############################################################################
## Save and Load
#############################################################################

def save_models_likelihoods(Models, Likelihoods, file_path):
    state_dicts = {
        'models': [model.state_dict() for model in Models],
        'likelihoods': [likelihood.state_dict() for likelihood in Likelihoods]
    }
    torch.save(state_dicts, file_path)


def load_models_likelihoods(file_path, model_class, likelihood_class, train_x, inducing_points, covar_type='RBF', device='cpu'):
    state_dicts = torch.load(file_path)
    
    Models = []
    Likelihoods = []
    for model_state, likelihood_state in zip(state_dicts['models'], state_dicts['likelihoods']):
        model = model_class(train_x, inducing_points=inducing_points, covar_type=covar_type)
        model.load_state_dict(model_state)
        model = model.to(device)
        
        likelihood = likelihood_class()
        likelihood.load_state_dict(likelihood_state)
        likelihood = likelihood.to(device)
        
        Models.append(model)
        Likelihoods.append(likelihood)
    
    return Models, Likelihoods

#################
##
################

def get_outlier_indices_iqr(data, outbound = 1.5):
    mask = np.ones(data.shape[0], dtype=bool)
    
    for i in range(data.shape[1]):
        Q1 = np.percentile(data[:, i], 25)
        Q3 = np.percentile(data[:, i], 75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - outbound * IQR
        upper_bound = Q3 + outbound * IQR
        
        mask = mask & (data[:, i] >= lower_bound) & (data[:, i] <= upper_bound)
    
    outlier_indices = np.where(~mask)[0]  
    return outlier_indices