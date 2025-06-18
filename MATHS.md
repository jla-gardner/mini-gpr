# A condensed summary of Gaussian Process Regression techniques

## Preliminaries

Assume we have:

- $X \in \mathbb{R}^{N \times d}$ : the locations of $N$ data points, each of dimension $d$
- $y \in \mathbb{R}^{N}$ : corresponding (noisy) observations of some unknown function $f$
- $k: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ : a kernel function defining the covariance between pairs of data points. 
- $T \in \mathbb{R}^{T \times d}$ : the locations of $T$ test points
- $\sigma_{\epsilon}^2$ : an estimate of the variance of the noise in the observations

and the following notation:

- $I_N$ : the $N \times N$ identity matrix
- $K_{AB} = K(A, B)$ : the $A \times B$ matrix of pairwise kernel evaluations between two sets of points $A$ and $B$

Given this setup, we can now define various GP regression models with predictive mean $\hat{y}(T)$ and predictive covariance $\text{cov}(y(T))$.

## Full GPR

We make use of all data points to make predictions: this gives rise to $O(N^3)$ time complexity, and $O(N^2)$ memory complexity.

The **predictive mean** is given by:
$$\hat{y}(T) = K_{TX} \left(K_{XX} + \sigma_{\epsilon}^2 I_N\right)^{-1} y$$

The **predictive covariance** is given by:
$$\text{cov}(y(T)) = K_{TT} - K_{TX} \left(K_{XX} + \sigma_{\epsilon}^2 I_N\right)^{-1} K_{XT}$$

The **log marginal likelihood** is given by:
$$
\log p(y|X) = -\frac{1}{2\sigma_{\epsilon}^2}(y - \hat{y}(X))^T (y - \hat{y}(X)) - \frac{1}{2\sigma_{\epsilon}^2}\log |K(X, X) + \sigma_{\epsilon}^2 I| - \frac{N}{2}\log 2\pi
$$


## SoR

We use a set of "inducing points", $M \in \mathbb{R}^{M \times d}$, to apprixmate the full covariance matrix. Note that these points are not necessarily from the training data.

The core idea of SoR is to base all predictions on a smaller, representative subset of the original training data, which you've denoted as inducing points $M \in \mathbb{R}^{M \times d}$. The model essentially behaves as if the *only* information it has comes from this subset. Unlike FITC, SoR does not retain any individual information about the training points that are *not* in the subset $M$. It uses the subset to learn a global model and then applies that model to the entire dataset.

Let's assume the "regressors" (the inducing points) are $M$ and the corresponding (often hypothetical or subset-selected) outputs are $u_m$. The SoR approximation effectively computes the posterior distribution of these inducing outputs, $p(u_m|y)$, and then uses them for all predictions.

The updated equations for the SoR predictive mean and covariance at test points $T \in \mathbb{R}^{T \times d}$ are as follows.

First, using the same shorthand for the kernel covariance matrices:
* $K_{mm} = K(M, M)$ is the $M \times M$ covariance matrix between the inducing points.
* $K_{mn} = K(M, X)$ is the $M \times N$ covariance matrix between the inducing and training points.
* $K_{nm} = K(X, M)$ is the $N \times M$ transpose of $K_{mn}$.
* $K_{tm} = K(T, M)$ is the $T \times M$ covariance matrix between the test and inducing points.

The **SoR predictive mean**, $\bar{f}_{*,\text{SoR}}$, is given by:
$$\bar{f}_{*,\text{SoR}} = K_{tm} (K_{mn}K_{nm} + \sigma_{\epsilon}^2 K_{mm})^{-1} K_{mn} y$$

The **SoR predictive covariance**, $\text{cov}(f_{*,\text{SoR}})$, is given by:
$$\text{cov}(f_{*,\text{SoR}}) = K(T, T) - K_{tm} K_{mm}^{-1} K_{mt} + \sigma_{\epsilon}^2 K_{tm} (K_{mn}K_{nm} + \sigma_{\epsilon}^2 K_{mm})^{-1} K_{mt}$$


---

# FITC

The Fully Independent Training Conditional (FITC) is a sparse approximation method for Gaussian Processes that is particularly useful when the number of training points, $N$, is large. It introduces a set of $M$ inducing points, where $M < N$, to create a low-rank approximation of the full covariance matrix.

The core idea behind FITC is to assume that the training data points are conditionally independent given the inducing points. This simplifies the model by making the approximate covariance matrix of the training outputs, $K(X, X)$, sparse.

Given your inducing point locations, $M \in \mathbb{R}^{M \times d}$, the updated equations for the predictive mean and covariance at test points $T \in \mathbb{R}^{T \times d}$ are as follows.

First, we define a shorthand notation for the kernel covariance matrices:
* $K_{mm} = K(M, M)$ is the $M \times M$ covariance matrix between the inducing points.
* $K_{mn} = K(M, X)$ is the $M \times N$ covariance matrix between the inducing and training points.
* $K_{nn} = K(X, X)$ is the $N \times N$ covariance matrix between the training points.
* $K_{tm} = K(T, M)$ is the $T \times M$ covariance matrix between the test and inducing points.

A key component in the FITC approximation is the diagonal matrix $\Lambda$:
$$\Lambda = \text{diag}[K_{nn} - K_{nm}K_{mm}^{-1}K_{mn}]$$
This $N \times N$ diagonal matrix captures the variance of each training point that is not explained by the inducing points. Let's denote its diagonal elements as $\Lambda_{ii}$.

The **FITC predictive mean**, $\bar{f}_{*,\text{FITC}}$, is given by:
$$\bar{f}_{*,\text{FITC}} = K_{tm} (K_{mm} + K_{mn}(\Lambda + \sigma_{\epsilon}^2 I)^{-1}K_{nm})^{-1} K_{mn} (\Lambda + \sigma_{\epsilon}^2 I)^{-1} y$$

The **FITC predictive covariance**, $\text{cov}(f_{*,\text{FITC}})$, is given by:
$$\text{cov}(f_{*,\text{FITC}}) = K(T, T) - K_{tm}K_{mm}^{-1}K_{mt} + K_{tm} (K_{mm} + K_{mn}(\Lambda + \sigma_{\epsilon}^2 I)^{-1}K_{nm})^{-1} K_{mt}$$
