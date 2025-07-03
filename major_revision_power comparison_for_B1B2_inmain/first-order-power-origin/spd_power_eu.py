import numpy as np
import sys
import time
from multiprocess import Pool
import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization

from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.matrices import Matrices, MatricesMetric

from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.geometry import spd_matrices

s1 = int(sys.argv[1])
s2 = int(sys.argv[2])

# define the manifold to be d*d spd matrix, d = 3
d = 3
mfd = spd_matrices.SPDMetricAffine(n=d, power_affine=1)
aux = spd_matrices.SPDMatrices(n=d)
tau_lst = [0.25,0.5,0.75,1,1.5,1.25,1.75]

T_lst=[50,100,500]
tau =tau_lst[s1]
nsamples = T_lst[s2]
print('tau:'+str(tau))
print('nsamples:'+str(nsamples))
mut =mfd.geodesic(gs.eye(d),end_point=2* gs.eye(d))
def generate_sample_tangent_space(base=gs.eye(d), n_samples=1, sigma=1):
    # generate i.i.d gaussian noise on tangent space
    size = (n_samples, d, d) if n_samples != 1 else (d, d)
    tangent_vec_at_id_aux = gs.random.normal(scale=sigma, size=size)
    tangent_vec_at_id = 0.5 * (tangent_vec_at_id_aux + Matrices.transpose(
        tangent_vec_at_id_aux
    ))
    # sqrt_base_point = gs.linalg.sqrtm(base_point)
    return tangent_vec_at_id


def generate_random_sample(base=gs.eye(d), n_samples=1, sigma=1):
    # generate sample with Frechet mean equal the given base point.
    tv = generate_sample_tangent_space(base=base, n_samples=n_samples, sigma=sigma)
    return mfd.exp(tangent_vec=tv, base_point=base)


def generate_ar_ts(base=gs.eye(d), n_samples=1, sigma=1, rho=0.5,tau=1,bt=2* gs.eye(d)):
    # generate AR(1)  process X_t -mu = rho(t/n)(X_{t-1}-mu)+epsilon
    ## rho(u) = 0.3+0.2*u^2
    
   
    if n_samples == 1:
        return generate_sample_tangent_space(base=base, n_samples=1, sigma=1)
    else:
        data_tv = np.zeros((n_samples, d, d))
        data= np.zeros((n_samples, d, d))
        data_tv[0, :, :] = generate_sample_tangent_space(base=base, n_samples=1, sigma=1)
        
        data[0] = mfd.exp(tangent_vec=data_tv[0],base_point=base)
        
        for i in range(1, n_samples):
            u = i / n_samples
            #             rhotmp =  rho+0.2*(u**2)
            
            #rhotmp = rho + 0.5*u*(1-u)
            delta = data_tv[i - 1, :, :]
            rhotmp = rho+0.25*u
            inten = ( (2.5*(u-0.25)) **2+0.2 )*sigma
            noise = generate_sample_tangent_space(base=gs.eye(d), n_samples=1, sigma=(inten/(1+3*tau)))
#            noise[0,0] =inten*noise[0,0]
#            noise[1,2] =inten*noise[1,2]
#            noise[2,1] =inten*noise[2,1]
#            noise[2,2] =inten*noise[2,2]
            if(tau>0):
                data_tv[i] = mfd.parallel_transport(tangent_vec=rhotmp * (delta) + noise, base_point=base, end_point=mut(tau*u))
                data[i] = mfd.exp(tangent_vec= data_tv[i], base_point=mut(tau*u))
            else:
                data_tv[i] = rhotmp * (delta) + noise
                                                    
                data[i] = mfd.exp(tangent_vec= data_tv[i], base_point=base)
       
        #data = mfd.exp(tangent_vec=data_tv, base_point=base)
        return data





def resvec_to_sum(res_vec, w=3):
    # local sum
    dim = res_vec.shape[1]
    N = res_vec.shape[0]
    res = np.zeros((N - w + 1, dim))
    for i in range(dim):
        res[:, i] = np.convolve(res_vec[:, i], np.ones(w), 'valid')

    return res


def gamma_m(res_vec, w=3):
    localvar = resvec_to_sum(res_vec, w) ** 2
    n, dim = localvar.shape
    res = np.zeros((n, dim))
    for i in range(dim):
        res[:, i] = np.cumsum(localvar[:, i]) / (w * n)
    return (res)


def resvec_to_sum(res_vec,w=3):
    # local sum
    dim = res_vec.shape[1]
    N = res_vec.shape[0]
    res =np.zeros((N-w+1,dim))
    for i in range(dim):
        res[:,i] = np.convolve(res_vec[:,i], np.ones(w), 'valid')
    
    return res

def gamma_m(res_vec,w=3):
    localvar = resvec_to_sum(res_vec,w)**2
    n,dim = localvar.shape
    res = np.zeros((n,dim))
    for i in range(dim):
        res[:,i] = np.cumsum(localvar[:,i])/(w*n)
    return(res)


def volatity(res_vec,wlst):
    wm =max(wlst)
    L = len(wlst)
    n,dim = res_vec.shape
    localvar_res=np.zeros(shape=(n-wm+1 ,dim,L))
    for i in range(L):
         localvar_res[:,:,i] = gamma_m(res_vec,wlst[i])[:(n-wm+1),]
    
    vol = np.zeros((n-wm+1, dim,L-2))
    for j in range(n-wm+1):
        for k in range(dim):
            for i in range(L-4):
                vol[j,k,i] = np.std(localvar_res[j,k,i:(i+3)])
    
    vol_sum = np.sum(vol,axis=1)
    
    return np.max(vol_sum,axis=0)

def select_window(res_vec,wlst):
    vol = volatity(res_vec,wlst)
    idx = np.argmin(np.max(vol,axis=0))
    return(wlst[idx+1])


def local_sum(res,w=3):
    # local sum for reisduals
    dim = res.shape[1]
    N = res.shape[0]
    locsum =np.zeros((N-w+1,dim))
    for i in range(dim):
        locsum[:,i] = np.convolve(res[:,i], np.ones(w), 'valid')
    
    return locsum

def generate_Phi(locsum):
    #Philst = np.zeros(locsum.shape)
    n = locsum.shape[0]
    g = np.random.normal(size=n)
    g = g.reshape((n,1))
    dPhi = g*locsum
    Phi = np.cumsum(dPhi,axis=0)
    return Phi


def bootstrap_test_mp(m):
    np.random.seed(m)
    data = generate_ar_ts(base=gs.eye(d) ,n_samples=nsamples,sigma=1,rho=0.05,tau=tau) #generate data , mean =id
    data =data.reshape((nsamples,9))
    #data=data[:,[0,1,2,4,5,8]]
    residual = data-np.mean(data,axis=0)
    res_cusum =np.cumsum(residual,axis=0)
    res_cusum_norm = np.sqrt(np.sum(res_cusum*res_cusum,axis=1))
    Tn = res_cusum_norm.max()/np.sqrt(nsamples)
    
    L =max(np.ceil(0.01*nsamples),2)
    if(nsamples<100):L=1
    U = max(0.05*nsamples+1,6)
    windows =np.arange(L,U)
    # windows =np.linspace(0.02, 0.05, num=20) * nsample= 400
    windows = (np.rint(windows)).astype(int)
    w = select_window(residual,windows) 
    B =2000# bootstrap size
    Boot_Stat= np.zeros(B)

    locsum = local_sum(residual,w=w)
    for i in range(B):
        #np.random.seed(seed*10000+i)
        Phi = generate_Phi(locsum)/np.sqrt(w*(nsamples-w+1))
        for k in range(nsamples-w+1):

            Phi[k] = Phi[k]-(k+1)/(nsamples-w+1)*Phi[nsamples-w]

        Phinorm = np.sqrt(np.sum(Phi*Phi,axis=1))

        Boot_Stat[i] = np.max(Phinorm[w:])
    pval =np.mean(Boot_Stat>=Tn) 
    return pval

    
##
#pp1 =np.zeros(20)
#pp2 =np.zeros(20)
#
#for k in range(20):
#    res=bootstrap_test_mp(k)
#    pp1[k] =res[0]
#    pp2[k] =res[1]
#print(np.mean(pp1<=0.1))
#print(np.mean(pp1))
#print(np.mean(pp2<=0.1))
#print(np.mean(pp2))

M =5000

pval = np.zeros(M)

#B = 3600 #bootstrap size
start1=time.time()
p=Pool(20)
#print(str(np.ceil(0.01*nsamples)))

try:
    result = p.map_async(bootstrap_test_mp, range(M))
    res = result.get()
except Exception as e:
    print(f"Error in multiprocessing: {e}")
    p.terminate()
    p.join()
    sys.exit(1)

for i in range(M):
    pval[i] = res[i]  # Ensure bootstrap_test_mp returns a single value

print(np.mean(pval <= 0.05))
np.save(f'spd_3_pval_eu_tau_{s1}_T_{s2}.npy', pval)

end1 = time.time()
print(end1 - start1)

p.close()
p.join()
