"""
File: Training.py
Author: Hongjin Ren
Description: Train the Gaussian process models

"""

#############################################################################
## Package imports
#############################################################################
import torch
import gpytorch
import tqdm as tqdm
import pandas as pd
import numpy as np

import GP_functions.GP_models as GP_models
import GP_functions.Loss_function as Loss_function
import GP_functions.Tools as Tools

from joblib import Parallel, delayed

#############################################################################
## Training LocalGP
#############################################################################

def train_one_column_StandardGP(local_train_x, local_train_y, covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=10, device='cpu'):

    local_train_x = local_train_x.to(device)
    local_train_y = local_train_y.to(device)

    # local_train_y_column = local_train_y[:, column_idx]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP_models.StandardGP(local_train_x, local_train_y, likelihood, covar_type)


    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    counter = 0
    iterator = tqdm.tqdm(range(num_iterations))

    for i in iterator:
    # for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(local_train_x)
        loss = -mll(output, local_train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if loss.item() <= best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model.load_state_dict(best_state)
                break

    return model, likelihood



def train_one_row_StandardGP(train_x, train_y, covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=10, device='cpu'):
    # Train the all columns of the output
    Models = []
    Likelihoods = []
    for column_idx in range(train_y.shape[1]):
        model, likelihood = train_one_column_StandardGP(train_x, train_y[:column_idx].squeeze(), covar_type, lr, num_iterations, patience, device)
        Models.append(model)
        Likelihoods.append(likelihood)
    return Models, Likelihoods


def train_one_row_StandardGP_Parallel(train_x, train_y, covar_type = 'RBF', lr=0.05, num_iterations=5000, patience=10, device='cpu'):
    # Helper function to train a single column
    def train_column(column_idx):
        model, likelihood = train_one_column_StandardGP(train_x, train_y[:column_idx].squeeze(), covar_type, lr, num_iterations, patience, device)
        return model, likelihood
    
    # Parallelize the training of all columns
    results = Parallel(n_jobs=17)(delayed(train_column)(column_idx) for column_idx in range(train_y.shape[1]))
    
    # Unzip the results
    Models, Likelihoods = zip(*results)
    return list(Models), list(Likelihoods)


