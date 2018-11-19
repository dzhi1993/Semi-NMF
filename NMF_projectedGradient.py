# NMF by alternative non-negative least squares using projected gradients
# Author: Chih-Jen Lin, National Taiwan University
# Python/numpy translation: Anthony Di Franco

from numpy import *
from numpy.linalg import norm
from time import time
from sys import stdout


def nmf(V, Winit, Hinit, tol, timelimit, maxiter):
    """
    (W,H) = nmf(V,Winit,Hinit,tol,timelimit,maxiter)
    W,H: output solution
    Winit,Hinit: initial solution
    tol: tolerance for a relative stopping condition
    timelimit, maxiter: limit of time and iterations
    """

    W = Winit;
    H = Hinit;
    initt = time();

    gradW = dot(W, dot(H, H.T)) - dot(V, H.T)
    gradH = dot(dot(W.T, W), H) - dot(W.T, V)
    initgrad = norm(r_[gradW, gradH.T])
    print('Init gradient norm %f' % initgrad)
    tolW = max(0.001, tol) * initgrad
    tolH = tolW

    for iter in range(1, maxiter):
        # stopping condition
        projnorm = norm(r_[gradW[logical_or(gradW < 0, W > 0)],
                           gradH[logical_or(gradH < 0, H > 0)]])
        if projnorm < tol * initgrad or time() - initt > timelimit: break

        (W, gradW, iterW) = nlssubprob(V.T, H.T, W.T, tolW, 1000)
        W = W.T
        gradW = gradW.T

        if iterW == 1: tolW = 0.1 * tolW

        (H, gradH, iterH) = nlssubprob(V, W, H, tolH, 1000)
        if iterH == 1: tolH = 0.1 * tolH

        if iter % 10 == 0: stdout.write('.')

    print ( '\nIter = %d Final proj-grad norm %f' % (iter, projnorm) )
    return (W, H)


def nlssubprob(V, W, Hinit, tol, maxiter):
    """
    H, grad: output solution and gradient
    iter: #iterations used
    V, W: constant matrices
    Hinit: initial solution
    tol: stopping tolerance
    maxiter: limit of iterations
    """

    H = Hinit
    WtV = dot(W.T, V)
    WtW = dot(W.T, W)

    alpha = 1;
    beta = 0.1;
    for iter in range(1, maxiter):
        grad = dot(WtW, H) - WtV
        projgrad = norm(grad[logical_or(grad < 0, H > 0)])
        if projgrad < tol: break

        # search step size
        for inner_iter in range(1, 20):
            Hn = H - alpha * grad
            Hn = where(Hn > 0, Hn, 0)
            d = Hn - H
            gradd = sum(grad * d)
            dQd = sum(dot(WtW, d) * d)
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0;
            if inner_iter == 1:
                decr_alpha = not suff_decr;
                Hp = H;
            if decr_alpha:
                if suff_decr:
                    H = Hn;
                    break;
                else:
                    alpha = alpha * beta;
            else:
                if not suff_decr or (Hp == Hn).all():
                    H = Hp;
                    break;
                else:
                    alpha = alpha / beta;
                    Hp = Hn;

        if iter == maxiter:
            print ('Max iter in nlssubprob')
    return (H, grad, iter)

w1 = array([[1,2,3],[4,5,6]])
h1 = array([[1,2],[3,4],[5,6]])
w2 = array([[1,1,3],[4,5,6]])
h2 = array([[1,1],[3,4],[5,6]])

v = dot(w1,h1)

(wo,ho) = nmf(v, w2, h2, 0.001, 10, 10)
print(wo)
print(ho)