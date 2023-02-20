"""
    Routines for computing matrix square roots.

    With ideas from:

    https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    https://github.com/pytorch/pytorch/issues/25481
"""

import pdb
import torch
from torch.autograd import Function
from functools import partial
import numpy as np
import scipy.linalg
try:
    import cupy as cp
except:
    import numpy as cp

#### VIA SVD, version 1: from https://github.com/pytorch/pytorch/issues/25481
def symsqrt_v1(A, func='symeig'):
    """Compute the square root of a symmetric positive definite matrix."""
    ## https://github.com/pytorch/pytorch/issues/25481#issuecomment-576493693
    ## perform the decomposition
    ## Recall that for Sym Real matrices, SVD, EVD coincide, |λ_i| = σ_i, so
    ## for PSD matrices, these are equal and coincide, so we can use either.
    if func == 'symeig':
        s, v = A.symeig(eigenvectors=True) # This is faster in GPU than CPU, fails gradcheck. See https://github.com/pytorch/pytorch/issues/30578
    elif func == 'svd':
        _, s, v = A.svd()                 # But this passes torch.autograd.gradcheck()
    else:
        raise ValueError()

    ## truncate small components
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


#### VIA SVD, version 2: from https://github.com/pytorch/pytorch/issues/25481
def symsqrt_v2(A, func='symeig'):
    """Compute the square root of a symmetric positive definite matrix."""
    if func == 'symeig':
        s, v = A.symeig(eigenvectors=True) # This is faster in GPU than CPU, fails gradcheck. See https://github.com/pytorch/pytorch/issues/30578
    elif func == 'svd':
        _, s, v = A.svd()                 # But this passes torch.autograd.gradcheck()
    else:
        raise ValueError()

    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps

    ### This doesn't work for batched version

    ### This does but fails gradcheck because of inpalce

    ### This seems to be equivalent to above, work for batch, and pass inplace. CHECK!!!!
    s = torch.where(above_cutoff, s, torch.zeros_like(s))

    sol =torch.matmul(torch.matmul(v,torch.diag_embed(s.sqrt(),dim1=-2,dim2=-1)),v.transpose(-2,-1))

    return sol

#
#

def special_sylvester(a, b):
    """Solves the eqation `A @ X + X @ A = B` for a positive definite `A`."""
    s, v = a.symeig(eigenvectors=True)
    d = s.unsqueeze(-1)
    d = d + d.transpose(-2, -1)
    vt = v.transpose(-2, -1)
    c = vt @ b @ v
    return v @ (c / d) @ vt


##### Via Newton-Schulz: based on
## https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py, and
## https://github.com/BorisMuzellec/EllipticalEmbeddings/blob/master/utils.py
def sqrtm_newton_schulz(A, numIters, reg=None, return_error=False, return_inverse=False):
    """ Matrix squareroot based on Newton-Schulz method """
    if A.ndim <= 2: # Non-batched mode
        A = A.unsqueeze(0)
        batched = False
    else:
        batched = True
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = (A**2).sum((-2,-1)).sqrt() # Slightly faster than : A.mul(A).sum((-2,-1)).sqrt()

    if reg:
        ## Renormalize so that the each matrix has a norm lesser than 1/reg,
        ## but only normalize when necessary
        normA *= reg
        renorm = torch.ones_like(normA)
        renorm[torch.where(normA > 1.0)] = normA[cp.where(normA > 1.0)]
    else:
        renorm = normA

    Y = A.div(renorm.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).to(A.device)#.type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).to(A.device)#.type(dtype)
    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA    = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    sAinv = Z/torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    if not batched:
        sA = sA[0,:,:]
        sAinv = sAinv[0,:,:]

    if not return_inverse and not return_error:
        return sA
    elif not return_inverse and return_error:
        return sA, compute_error(A, sA)
    elif return_inverse and not return_error:
        return sA,sAinv
    else:
        return sA, sAinv, compute_error(A, sA)

def create_symm_matrix(batchSize, dim, numPts=20, tau=1.0, dtype=torch.float32,
    verbose=False):
    """ Creates a random PSD matrix """
    A = torch.zeros(batchSize, dim, dim).type(dtype)
    for i in range(batchSize):
        pts = np.random.randn(numPts, dim).astype(np.float32)
        sA = np.dot(pts.T, pts)/numPts + tau*np.eye(dim).astype(np.float32);
        A[i,:,:] = torch.from_numpy(sA);
    if verbose: print('Creating batch %d, dim %d, pts %d, tau %f, dtype %s' % (batchSize, dim, numPts, tau, dtype))
    return A

def compute_error(A, sA):
    """ Computes error in approximation """
    normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1),dim=1))
    error = A - torch.bmm(sA, sA)
    error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
    return torch.mean(error)

###==========================

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: square root is not differentiable for matrices with zero eigenvalues.

    """
    @staticmethod
    def forward(ctx, input, method = 'numpy'):
        _dev = input.device
        if method == 'numpy':
            m = input.cpu().detach().numpy().astype(np.float_)
            sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).type_as(input)
        elif method == 'pytorch':
            sqrtm = symsqrt(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output, method = 'numpy'):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            if method == 'numpy':
                sqrtm = sqrtm.data.numpy().astype(np.float_)
                gm = grad_output.data.numpy().astype(np.float_)
                grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
                grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
            elif method == 'pytorch':
                grad_input = special_sylvester(sqrtm, grad_output)
        return grad_input


## ========================================================================== ##
## NOTE: Must pick which version of matrix square root to use!!!!

## sqrtm = MatrixSquareRoot.apply
sqrtm = symsqrt_v2
## sqrtm = symsqrt_v1
## sqrtm = symsqrt_diff
## ========================================================================== ##

def main():
    from torch.autograd import gradcheck

    k = torch.randn(5, 20, 20).double()
    M = k @ k.transpose(-1,-2)

    s1 = symsqrt_v1(M, func='symeig')
    test = torch.allclose(M, s1 @ s1.transpose(-1,-2))
    print('Via symeig:', test)

    s2 = symsqrt_v1(M, func='svd')
    test = torch.allclose(M, s2 @ s2.transpose(-1,-2))
    print('Via svd:  ', test)

    print('Sqrtm with symeig and svd match:', torch.allclose(s1,s2))

    M.requires_grad = True

    ## Check gradients for symsqrt
    _sqrt = partial(symsqrt, func='svd')
    test = gradcheck(_sqrt, (M,))
    print('Grach Check for sqrtm/svd:', test)

    ## Check symeig itself
    S = torch.rand(5,20,20, requires_grad=True).double()
    def func(S):
        x = 0.5 * (S + S.transpose(-2, -1))
        return torch.symeig(x, eigenvectors=True)
    print('Grad check for symeig', gradcheck(func, [S]))

    ## Check gradients for symsqrt with symeig
    _sqrt = partial(symsqrt, func='symeig')
    test = gradcheck(_sqrt, (M,))
    print('Grach Check for sqrtm/symeig:', test)

if __name__ == '__main__':
    main()
