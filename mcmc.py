import torch
import gpytorch
import pandas as pd
import numpy as np
import tqdm as tqdm
from linear_operator import settings

import pyro
import math
import pickle
import time
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az
import seaborn as sns

import GP_functions.Loss_function as Loss_function
import GP_functions.bound as bound
import GP_functions.Estimation as Estimation
import GP_functions.Training as Training
import GP_functions.Prediction as Prediction
import GP_functions.GP_models as GP_models
import GP_functions.Tools as Tools

Device = 'cuda'

TrainData = pd.read_csv('Data/train3D.csv', delimiter=',')
TestData = pd.read_csv('Data/test3D.csv', delimiter=',')




scaler = StandardScaler()
TrainData_standardized = pd.DataFrame(scaler.fit_transform(TrainData), columns=TrainData.columns).values

TestData_standardized = pd.DataFrame(scaler.fit_transform(TestData), columns=TestData.columns).values

X_train = TrainData_standardized[:,0:2]
Y_train = TrainData_standardized[:,-1]

X_test = TestData_standardized[:,0:2]
Y_test = TestData_standardized[:,-1]




train_x = torch.tensor(X_train, dtype=torch.float32).to(Device)
test_x = torch.tensor(X_test, dtype=torch.float32).to(Device)

train_y = torch.tensor(Y_train, dtype=torch.float32).to(Device)
test_y = torch.tensor(Y_test, dtype=torch.float32).to(Device)


LocalGP_models, LocalGP_likelihoods = Training.train_one_column_StandardGP(
    train_x, train_y, covar_type='RBF', lr=0.05, num_iterations=5000, patience=10, device=Device, use_amp=False
    )




if __name__ == '__main__':
    row_idx = 0

    bounds = bound.get_bounds(train_x)

    mcmc_result = Estimation.run_mcmc(Prediction.preds_for_one_model, LocalGP_models, LocalGP_likelihoods, row_idx, test_y, bounds, 
                                                  num_sampling = 4, warmup_step = 1, num_chains=2, device=Device,jit_compile=False)
    samples = mcmc_result.get_samples()
    print("theta 后验样本：", samples.get('theta'))
    print("sigma 后验样本：", samples.get('sigma'))