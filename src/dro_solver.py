"""Exact discrete Wasserstein-1 worst-case expectation for finite empirical support.

For empirical support z_i with nominal weights p_i and utility u_i, solve
min_q,T sum_j q_j u_j s.t. transport from p to q has cost <= epsilon.
This is a finite LP and is suitable for auditable benchmark-scale supports.
"""
import numpy as np
from scipy.optimize import linprog

def wasserstein_worst_case(utilities, distances, epsilon, weights=None):
    u=np.asarray(utilities,dtype=float); D=np.asarray(distances,dtype=float); n=len(u)
    if D.shape!=(n,n): raise ValueError('distances must be n x n')
    p=np.ones(n)/n if weights is None else np.asarray(weights,dtype=float)
    if not np.isclose(p.sum(),1): p=p/p.sum()
    # Variables T_ij. q_j = sum_i T_ij. Objective sum_ij T_ij*u_j.
    c=np.tile(u,n)
    Aeq=[]; beq=[]
    for i in range(n):
        row=np.zeros(n*n); row[i*n:(i+1)*n]=1; Aeq.append(row); beq.append(p[i])
    Aub=np.asarray(D).reshape(1,-1); bub=[epsilon]
    res=linprog(c,A_ub=Aub,b_ub=bub,A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=(0,None),method='highs')
    if not res.success: raise RuntimeError(res.message)
    return float(res.fun), res.x.reshape(n,n)
