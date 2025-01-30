"""
File: GP_models.py
Author: Hongjin Ren
Description: Various Gaussian process models based on GpyTorch

"""

#############################################################################
## Package imports
#############################################################################

import torch
import gpytorch


#############################################################################
## Set up the model structure (LocalGP, SparseGP, MultitaskGP)
#############################################################################

## LocalGP

class StandardGP(gpytorch.models.ExactGP):
    # get the training data and likelihoods, and construct any objects needed for the model's forward methods.
    def __init__(self, train_x, train_y, likelihood, covar_type = 'RBF'):
        super(StandardGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if covar_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)))
        elif covar_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1)))
        elif covar_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1)))
        elif covar_type == 'RQ':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=train_x.size(-1)))
        elif covar_type == 'PiecewisePolynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(q=2, ard_num_dims=train_x.size(-1)))
        else:
            print('You should choose one of these kernels (RBF, Matern5/2, Matern3/2, RQ, PiecewisePolynomial)')
        

    def forward(self, x):
        # denotes the a priori mean and covariance matrix of GP
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    



# class BatchIndependentLocalGP(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(BatchIndependentLocalGP, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y.shape[1]]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1), batch_shape=torch.Size([train_y.shape[1]])),
#             batch_shape=torch.Size([train_y.shape[1]])
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
#             gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#         )


