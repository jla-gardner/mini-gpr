The log marginal likelihood, often referred to as the "model evidence," is a crucial quantity in Gaussian Process models. It is used to optimize the kernel's hyperparameters (e.g., lengthscale, signal variance) and the noise variance $\sigma_{\epsilon}^2$ by finding the values that maximize this likelihood.

Here are the equations for the log marginal likelihood for the full GP, FITC, and SoR approaches.

### 1. Full Gaussian Process Regression

For the exact, full GP model, the log marginal likelihood, $\log p(y|X)$, is given by:

$$\log p(y|X) = -\frac{1}{2} y^T (K_{nn} + \sigma_{\epsilon}^2 I)^{-1} y - \frac{1}{2} \log |K_{nn} + \sigma_{\epsilon}^2 I| - \frac{N}{2} \log(2\pi)$$

* **Data Fit Term**: $-\frac{1}{2} y^T (K_{nn} + \sigma_{\epsilon}^2 I)^{-1} y$
    This term measures how well the model fits the training data. A good fit results in a smaller negative value.
* **Complexity Penalty Term**: $-\frac{1}{2} \log |K_{nn} + \sigma_{\epsilon}^2 I|$
    This term is the log determinant of the covariance matrix. It penalizes model complexity; overly complex models (e.g., those with very high variance or short lengthscales) will have a larger determinant, which makes this term more negative.
* **Normalization Constant**: $-\frac{N}{2} \log(2\pi)$
    This is a constant that doesn't depend on the model hyperparameters, so it is often omitted during optimization.

Here, $K_{nn} = K(X, X)$ is the $N \times N$ covariance matrix of the training data.

---

### 2. Fully Independent Training Conditional (FITC)

For the FITC approximation, we don't have the exact marginal likelihood. Instead, we maximize a lower bound on it, often called the Evidence Lower Bound (ELBO). The FITC approximate log marginal likelihood is given by:

$$\log p(y|X)_{\text{FITC}} \approx -\frac{1}{2} y^T (\Lambda + \sigma_{\epsilon}^2 I + K_{nm}K_{mm}^{-1}K_{mn})^{-1} y - \frac{1}{2} \log |\Lambda + \sigma_{\epsilon}^2 I + K_{nm}K_{mm}^{-1}K_{mn}| - \frac{N}{2} \log(2\pi)$$

While the above form is correct, a more numerically stable and insightful form is:

$$\log p(y|X)_{\text{FITC}} = -\frac{1}{2} y^T \Sigma_{\text{FITC}}^{-1} y - \frac{1}{2} \log |\Sigma_{\text{FITC}}| - \frac{N}{2} \log(2\pi)$$
where $\Sigma_{\text{FITC}} = Q_{nn} + \sigma_{\epsilon}^2I$ and $Q_{nn} = K_{nm}K_{mm}^{-1}K_{mn} + \Lambda$.

This is composed of:
* A standard Gaussian log-likelihood term, but with the full covariance $K_{nn}$ replaced by its low-rank plus diagonal approximation $Q_{nn}$.
* The log determinant term, $\log|\Sigma_{\text{FITC}}|$, which can be computed efficiently as:
    $$\log|\Sigma_{\text{FITC}}| = \log|K_{mm}| + \log|\sigma_{\epsilon}^{-2}K_{mn}K_{nm} + K_{mm}| + \sum_{i=1}^N \log(\Lambda_{ii} + \sigma_{\epsilon}^2)$$

The key components are:
* $K_{mm} = K(M, M)$, $K_{mn} = K(M, X)$, $K_{nm} = K(X, M)$
* $\Lambda = \text{diag}[K_{nn} - K_{nm}K_{mm}^{-1}K_{mn}]$

---

### 3. Subset of Regressors (SoR)

The SoR log marginal likelihood is also an approximation. It is derived by assuming the predictive distribution is based entirely on the inducing points.

$$\log p(y|X)_{\text{SoR}} = -\frac{1}{2} y^T \Sigma_{\text{SoR}}^{-1} y - \frac{1}{2} \log |\Sigma_{\text{SoR}}| - \frac{N}{2} \log(2\pi)$$
where the covariance matrix $\Sigma_{\text{SoR}}$ is defined as:
$$\Sigma_{\text{SoR}} = K_{nm}K_{mm}^{-1}K_{mn} + \sigma_{\epsilon}^2 I$$

* **Data Fit Term**: The data fit relies on the covariance projected through the inducing points, $K_{nm}K_{mm}^{-1}K_{mn}$.
* **Complexity Penalty Term**: The log determinant, $\log|\Sigma_{\text{SoR}}|$, also depends on this low-rank approximation. This term can also be computed efficiently using the matrix determinant lemma, avoiding the need to form the full $N \times N$ matrix.

Notice the key difference from FITC: the SoR likelihood **completely ignores the diagonal correction term $\Lambda$**. This reflects the fact that SoR discards the individual variances of the training points not captured by the inducing set, which often leads to poorer hyperparameter learning and an underestimation of model uncertainty compared to FITC.