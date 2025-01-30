"""
File: Estimation.py
Author: Hongjin Ren
Description: Train the Gaussian process models

"""

#############################################################################
## Package imports
#############################################################################
import torch
import numpy as np
import GP_functions.Loss_function as Loss_function
from scipy.optimize import basinhopping
import GP_functions.Prediction as Prediction
import tqdm

import scipy.stats as stats

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as transforms
from pyro.infer import MCMC, NUTS
import arviz as az

#############################################################################
## 
#############################################################################


def estimate_params_basinhopping_NM(models, likelihoods, row_idx, test_y, bounds):

    # Use basinhopping to estimate parameters for the GP model.

    def surrogate_loss_wrapped(params):
        return Loss_function.surrogate_loss_euclid(params, models, likelihoods, row_idx, test_y)


    # Define the bounds in the minimizer_kwargs
    minimizer_kwargs = {"method": "Nelder-Mead", 
                        "bounds": bounds,
                        "options": {"adaptive": True}}

    # Initialize the starting point
    initial_guess = [np.mean([b[0], b[1]]) for b in bounds]

    # Run basinhopping
    result = basinhopping(surrogate_loss_wrapped, initial_guess, minimizer_kwargs=minimizer_kwargs, 
                          niter=100, T = 1e-05, stepsize=0.25, niter_success = 20, target_accept_rate = 0.6)
    
    return result.x, result.fun




def estimate_params_for_one_model_Adam(model, likelihood, row_idx, test_y, initial_guess, param_ranges, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.5, device='cpu'):
    
    target_y = test_y[row_idx].to(device)
    target_x = torch.tensor(initial_guess, dtype=torch.float32).to(device).unsqueeze(0).requires_grad_(True)
    
    optimizer = torch.optim.Adam([target_x], lr=lr)
    
    model.eval()
    likelihood.eval()

    best_loss = float('inf')
    counter = 0

    iterator = tqdm.tqdm(range(num_iterations))

    for i in iterator:
    # for i in range(num_iterations):
        optimizer.zero_grad()
        
        loss = torch.norm(likelihood.to(device)(model.to(device)(target_x)).mean - target_y, p=2).sum()
        loss.backward(retain_graph=True)
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

        grad_norm = target_x.grad.data.norm(2).item()
        if grad_norm < attraction_threshold:
            target_x.grad.data += repulsion_strength * torch.randn_like(target_x.grad.data)
            optimizer.step()

        with torch.no_grad():
            for idx, (min_val, max_val) in enumerate(param_ranges):
                target_x[0, idx] = target_x[0, idx].clamp(min_val, max_val)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = target_x.detach().clone()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Stopping early due to lack of improvement.")
                target_x = best_state
                break
    
    return target_x.squeeze(), best_loss







def multi_start_estimation(model, likelihood, row_idx, test_y, param_ranges, estimate_function, num_starts=5, num_iterations=1000, lr=0.05, patience=50, attraction_threshold=0.1, repulsion_strength=0.1, device='cpu'):
    best_overall_loss = float('inf')
    best_overall_state = None

    quantiles = np.linspace(0.25, 0.75, num_starts)  
    
    for start in range(num_starts):
        print(f"Starting optimization run {start+1}/{num_starts}")
        
        initial_guess = [np.quantile([min_val, max_val], quantiles[start]) for (min_val, max_val) in param_ranges]

        estimated_params, loss = estimate_function(
            model, likelihood, row_idx, test_y, initial_guess, param_ranges,
            num_iterations=num_iterations, lr=lr, patience=patience,
            attraction_threshold=attraction_threshold, repulsion_strength=repulsion_strength, device=device
        )

        if loss < best_overall_loss:
            best_overall_loss = loss
            best_overall_state = estimated_params

    return best_overall_state.detach().numpy(), best_overall_loss



def run_mcmc(Pre_function, Models, Likelihoods, row_idx, test_y, bounds, num_sampling=2000, warmup_step=1000, num_chains=1):
    def model():
        params = []
        
        for i, (a, b) in enumerate(bounds):
            base_dist = dist.Normal(0, 1)
            transform = transforms.ComposeTransform([
                transforms.SigmoidTransform(),
                transforms.AffineTransform(loc=a, scale=b - a)
            ])
            transformed_dist = dist.TransformedDistribution(base_dist, transform)
            
            param_i = pyro.sample(f'param_{i}', transformed_dist)
            params.append(param_i)
        
        theta = torch.stack(params)
        
        sigma = pyro.sample('sigma', dist.HalfNormal(10.0))

        mu_value = Pre_function(Models, Likelihoods, theta.unsqueeze(0)).squeeze()

        y_obs = test_y[row_idx]
        
        pyro.sample('obs', dist.Normal(mu_value, sigma), obs=y_obs)


    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_sampling, warmup_steps=warmup_step, num_chains=num_chains)
    mcmc.run()

    return mcmc