################################################################################
############### COLLECTION OF FUNCTIONALS ON DATASETS ##########################
################################################################################
import numpy as np
import torch

class Functional():
    """
        Defines a JKO functional over measures implicitly by defining it over
        individual particles (points).

        The input should be a full dataset: points X (n x d) with labels Y (n x 1).
        Optionally, the means/variances associated with each class can be passed.

        (extra space do to repeating)

    """
    def __init__(self, V=None, W=None, f=None, weights=None):
        self.V = V # The functional on Z space in potential energy 𝒱() = V
        self.W = W # The bi-linear form on ZxZ spaces in interaction energy 𝒲
        self.f = f # The scalar-valued function in the niternal energy term ℱ

    def __call__(x, y, μ=None, Σ=None):
        sum = 0
        if self.F is not None:
            sum += self.F(x,y,μ,Σ)
        if self.V is not None:
            sum += self.V(x,y,μ,Σ)
        if self.W is not None:
            sum += self.W(x,y,μ,Σ)
        return sum

################################################################################
#######    Potential energy functionals (denoted by V in the paper)    #########
################################################################################

def affine_feature_norm(X,Y=None,A=None, b=None, threshold=None, weight=1.0):
    """ A simple (feature-only) potential energy based on affine transform + norm:

            v(x,y) = || Ax - b ||, so that V(ρ) = ∫|| Ax - b || dρ(x,y)

        where the integral is approximated by empirical expectation (mean).
    """
    if A is None and b is None:
        norm = X.norm(dim=1)
    elif A is None and not b is None:
        norm = (X - b).norm(dim=1)
    elif not A is None and b is None:
        norm = (X - b).norm(dim=1)
    else:
        norm = (X@A - b).norm(dim=1)
    if threshold:
        norm = torch.nn.functional.threshold(norm, threshold, 0)
    return weight*norm.mean()

def binary_hyperplane_margin(X, Y, w, b, weight=1.0):
    """ A potential function based on margin separation according to a (given
        and fixed) hyperplane:

        v(x,y) = max(0, 1 - y(x'w - b) ), so that V(ρ) = ∫ max(0, y(x'w - b) ) dρ(x,y)

    Returns 0 if all points are at least 1 away from margin.

    Note that y is expected to be {0,1}

    Needs separation hyperplane be determined by (w, b) parameters.
    """
    Y_hat = 2*Y-1 # To map Y to {-1, 1}, required by the SVM-type margin obj we use
    margin = torch.relu(1-Y_hat*(torch.matmul(X, w) - b))
    return weight*margin.mean()

def dimension_collapse(X, Y, dim=1, v=None, weight=1.0):
    """ Potential function to induce a dimension collapse """
    if v is None:
        v = 0
    deviation = (X[:,dim] - v)**2
    return weight*deviation.mean()



def cluster_repulsion(X, Y):
    pdb.set_trace()

################################################################################
########    Interaction energy functionals (denoted by W in the paper) #########
################################################################################

def interaction_fun(X, Y, weight=1.0):
    """

    """
    Z = torch.cat((X, Y.float().unsqueeze(1)), -1)

    n,d = Z.shape
    Diffs = Z.repeat(n,1,1).transpose(0,1) - Z.repeat(n,1,1)

    def _f(δz): # Enforces cluster repulsion:
        δx, δy = torch.split(δz,[δz.shape[-1]-1,1], dim=-1)
        δy = torch.abs(δy/δy.max()).ceil() # Hacky way to get 0/1 loss for δy
        return -(δx*δy).norm(dim=-1).mean(dim=-1)

    val = _f(Diffs).mean()

    return val*weight


def binary_cluster_margin(X, Y, μ=None, weight=1.0):
    """ Similar to binary_hyperplane_margin but does to require a separating
    hyperplane be provided in advance. Instead, computes one based on current
    datapoints as the hyperplane through the midpoint of their means.

    Also, ensures that ..., so it requires point-to-point comparison (interaction)

    """

    μ_0 = X[Y==0].mean(0)
    μ_1 = X[Y==1].mean(0)

    n,d = X.shape
    diffs_x = X.repeat(n,1,1).transpose(0,1) - X.repeat(n,1,1)
    diffs_x = torch.nn.functional.normalize(diffs_x, dim=2, p=2)

    μ = torch.zeros(n,d)
    μ[Y==0,:] = μ_0
    μ[Y==1,:] = μ_1

    diffs_μ = μ.repeat(n,1,1).transpose(0,1) - μ.repeat(n,1,1)
    diffs_μ = torch.nn.functional.normalize(diffs_μ, dim=2, p=2)


    inner_prod = torch.einsum("ijk,ijl->ij", diffs_x, diffs_μ)

    print(inner_prod.min(), inner_prod.max())

    out = torch.relu(-inner_prod + 1)

    print(out.shape)

    margin = torch.exp(out)
    return weight*margin.mean()
