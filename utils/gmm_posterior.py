from scipy.stats import multivariate_normal
import numpy as np

def gmm_posterior(x, means=None, covariances=None, weights=None):
    if means == None:
        means = np.array([
            [0, 0, 0],
            [3, 3, 3],
            [0, -3, -3],
            [3, 0, 0]
            ]) 
    if covariances == None:
        covariances = np.array([
            3 * np.eye(3),
            1.5 * np.eye(3),
            3 * np.eye(3),
            4 * np.eye(3)
            ])
    if weights == None:
        weights = np.array([0.1, 0.3, 0.5, 0.1])

    n_components = len(weights)
    n_features = x.shape[0]
    
    # Likelihood
    likelihoods = np.array([multivariate_normal.pdf(x, mean=means[i], cov=covariances[i]) for i in range(n_components)])
    
    # P(X=x | Y=i) * P(Y=i)
    unnormalized_posteriors = weights * likelihoods
    
    # Posterior
    posterior = unnormalized_posteriors / np.sum(unnormalized_posteriors)
    
    return posterior


