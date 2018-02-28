from sklearn.kernel_approximation import RBFSampler,Nystroem
import scipy.linalg as la
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics.pairwise import rbf_kernel
from params import ts_depths,n_fea,gamma, np, sp
class Whitener:
    def __init__(self,X):
        self.Xmean = X.mean(0)
        self.Xstd = X.std(0)
    def whiten(self,Z):
        return (Z-self.Xmean)/self.Xstd
    def unwhiten(self,Zw):
        return Zw*self.Xstd + self.Xmean

def expkern(x,y):
    return np.exp(-gamma*la.norm(x-y))

wh = Whitener(ts_depths)
ts_depths_w = wh.whiten(ts_depths)
xx = np.linspace(ts_depths_w.min(),ts_depths_w.max(),n_fea)[:,np.newaxis]
rbf_tr = Nystroem(expkern,gamma,n_components=n_fea)
#rbf_tr = Nystroem(gamma=gamma,n_components=n_fea)
#class rbf_transformer:
#    def __init__(self,X,gamma):
#        self.X = X
#        self.gamma = gamma
#    def transform(self,xx):
#        return rbf_kernel(xx,self.X) 

#rbf_tr = rbf_transformer(xx,gamma)
#rbf_tr.fit(x)
rbf_tr.fit(xx)
ts_depths_tr = rbf_tr.transform(ts_depths_w)

