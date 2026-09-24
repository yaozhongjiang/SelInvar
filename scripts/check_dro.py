import sys; sys.path.insert(0, __file__.rsplit("/",2)[0])
from src.dro_solver import wasserstein_worst_case
import numpy as np
u=np.array([1.,2.,5.]); D=np.abs(np.arange(3)[:,None]-np.arange(3)[None,:])
v,T=wasserstein_worst_case(u,D,epsilon=.5)
assert v <= u.mean()+1e-9
print('DRO LP sanity:',v)
