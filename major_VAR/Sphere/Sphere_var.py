import numpy as np
import sys
import time
from multiprocess import Pool
import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization



from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.matrices import Matrices, MatricesMetric

s1 = int(sys.argv[1])
s2 = int(sys.argv[2])

# define the manifold to be d*d spd matrix, d = 3

# define the manifold to be d*d spd matrix, d = 2
d=6
from geomstats.geometry.hypersphere import Hypersphere

sphere = Hypersphere(dim=d)
mfd= sphere.metric


tau_lst = [0,0.25,0.5,0.75,1.0,0.125,0.375]

cpunum= 8
T_lst=[50,100,500]
tau =tau_lst[s1]
nsamples = T_lst[s2]
print('tau:'+str(tau))
print('nsamples:'+str(nsamples))
print('numcpu:'+str(cpunum))

bp0 = np.array([0,0,0,0,0,0,1])

def generate_sample_tangent_space(base=bp0,n_samples=1,sigma=1):
    # generate i.i.d gaussian noise on tangent space
    size = (n_samples, d) if n_samples != 1 else (1,d)
    
    tangent_vec_aux = sigma*(gs.random.uniform(size=size)-0.5)
    tangent_vec = np.zeros(shape=(n_samples, d+1))
    tangent_vec[:,:d]=tangent_vec_aux
    #sqrt_base_point = gs.linalg.sqrtm(base_point)
    return tangent_vec


def generate_random_sample(base=bp0,n_samples=1,sigma=1):
    # generate sample with Frechet mean equal the given base point.
    tv =  generate_sample_tangent_space(base=base,n_samples=n_samples,sigma=sigma)
    return mfd.exp(tangent_vec=tv,base_point=base)

def generate_ar_ts(base=bp0,n_samples=1,sigma=1,rho=0.5,tau=1,bt=np.array([1,0,0,0,0,0,0]) ):
    # generate AR(1)  process X_t -mu = rho(t/n)(X_{t-1}-mu)+epsilon
    ## rho(u) = 0.3+0.2*u^2
    mut0 =mfd.geodesic(initial_point=base,end_point=bt)
    
    if n_samples ==1:
        return generate_random_sample(base,n_samples=1)
    else:
        data_tv =np.zeros((n_samples, d+1))
        data=np.zeros((n_samples, d+1))
        data_tv[0,:] = generate_sample_tangent_space(base=base,n_samples=1,sigma=sigma)
        data[0] = mfd.exp(tangent_vec=data_tv[0],base_point=base)
        for i in range(1,n_samples):
            u = i/n_samples
            #rhotmp =  0.2*np.cos(2*u*np.pi)+rho+u*(1-u)
            rhotmp = rho+0.5*u*(1-u)
            
            delta = data_tv[i-1,:]
            a = np.full(shape=(d+1,), fill_value=rhotmp)
            a[0]=1.1*rhotmp
            a[1]=0.9*rhotmp
            a[2]=0.8*rhotmp
            inten = 1.1*(1+u)*sigma/(1+tau)
            
            noise = generate_sample_tangent_space(base=base,n_samples=1,sigma=sigma/(1+tau))
            noise[0:3] =  inten*noise[0:3]
           # data_tv[i,:]  = rhotmp*delta + noise
            if(tau>0):
                data_tv[i, :] = mfd.parallel_transport(tangent_vec=a * (delta) + noise, base_point=base,
                                                    end_point=mut0(tau*u))
                data[i] = mfd.exp(tangent_vec= data_tv[i, :], base_point=mut0(tau*u))
            else:
                data_tv[i, :] = a * (delta) + noise
                                                    
                data[i] = mfd.exp(tangent_vec= data_tv[i, :], base_point=base)
                
        #tvnorm =mfd.inner_coproduct(data_tv,data_tv,base)
       # data =mfd.exp(tangent_vec=data_tv,base_point=base)
        return(data)


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

def Hess(res,mu):
    theta= np.sqrt(mfd.inner_product(res,res,mu))
    u = res/theta
    ftheta=theta/np.sin(theta)
    u = np.reshape(u,(d+1,1))
    x = np.reshape(mu,(d+1,1))
    
    
    H = np.dot(u,np.transpose(u)) + ftheta*np.cos(theta)* (
        np.eye(7)-np.dot(u,np.transpose(u))-np.dot(x,np.transpose(x)))
    return(H)
    

def Hprocess(residual,mean):
    N = residual.shape[0]
    h = np.zeros((N,7,7))
    for i in range(N):
        h[i,:,:] = Hess(residual[i,:],mean)/N
    H = np.cumsum(h,axis=0)
    return(H)

def Hinv(y,mean,H):
    l,v= np.linalg.eig(H)
    lidx = [0,1,2,3,4,5,6]
    lidx.remove(np.argmin(np.abs(l)))
    l = l[lidx]
    v = v[:,lidx]
    invvec = np.zeros(7)
    #invvec
    for k in range(6):
        invvec = mfd.inner_product(y,v[:,k])*v[:,k]/l[k] + invvec
        #mfd.inner_product(y,v[:,1])*v[:,1]/l[1]+mfd.inner_product(y,v[:,0])*v[:,0]/l[0]
    return(invvec)


def generate_Phi(locsum):
    #Philst = np.zeros(locsum.shape)
    n = locsum.shape[0]
    g = np.random.normal(size=n)
    g = g.reshape((n,1))
    dPhi = g*locsum
    Phi = np.cumsum(dPhi,axis=0)
    return Phi


def bootstrap_test(residual, mean, w=24 ,B=400,seed=2023):
    n =residual.shape[0]
    res_cusum =np.cumsum(residual,axis=0)
    Ht = Hprocess(residual,mean)

    res_cusum_norm = np.sqrt(mfd.inner_product(res_cusum,res_cusum,mean))

    Tn = res_cusum_norm.max()/np.sqrt(n)
    
    B =B# bootstrap size
    Boot_Stat= np.zeros(B)
    Boot_Stat2= np.zeros(B)
    locsum = local_sum(residual,w=w)
    for i in range(B):
        #np.random.seed(seed*10000+i)
        Phi = generate_Phi(locsum)/np.sqrt(w*(n-w+1))
        Phi2 = generate_Phi(locsum)/np.sqrt(w*(n-w+1))
        HinvPhi = Hinv(Phi[n-w],mean,Ht[n-w])
        
        for k in range(n-w+1):
            Phi[k] = Phi[k]-np.dot(Ht[k],HinvPhi)
            Phi2[k] = Phi2[k]-(k+1)/(n-w+1)*Phi2[n-w]
        
        Phinorm = np.sqrt(mfd.inner_product(Phi,Phi,mean))
        Phinorm2 = np.sqrt(mfd.inner_product(Phi2,Phi2,mean))
        Boot_Stat[i] = np.max(Phinorm[w:])
        Boot_Stat2[i] = np.max(Phinorm2[w:])
        
    return np.mean(Boot_Stat>=Tn), np.mean(Boot_Stat2>=Tn)


def bootstrap_test_mp(m):
    
    np.random.seed(m)
    
#     pval1 = np.zeros(M)
#     pval2 = np.zeros(M)
    B = 2000 #bootstrap size
    bp = bp0
    
    data = generate_ar_ts(base=bp0 ,n_samples=nsamples,sigma=1,rho=0.05,tau=tau) #generate data , mean =id

    L = max(0.02*nsamples,2)
    U = 0.1*nsamples+1
    windows =np.arange(L,U)
   # windows =np.linspace(0.02, 0.05, num=20) * nsample= 400
    windows = (np.rint(windows)).astype(int)
    #windows= (np.rint(windows)).astype(int)# window size candidate
    
    
    fmean =  FrechetMean(metric=mfd,epsilon=0.0000001,max_iter=100000)
    fmean.fit(data) # frechet mean function
    mean =fmean.estimate_ # empirical frechet mean
    
    residual = mfd.log(point=data,base_point=mean) # residual
 
    w1 = select_window(residual,windows) # selec window size in bootstrap
    
    
    pval1,pval2 = bootstrap_test(residual, mean, w=w1 ,B=B,seed=m)

    return (pval1, pval2,w1)




M =5000
#nsamples= 1000
pval1 = np.zeros(M)
pval2 = np.zeros(M)
#B = 3600 #bootstrap size
start1=time.time()
p=Pool(12)

result = p.map_async(bootstrap_test_mp, range(M))


#result.get()
res = result.get()
#
pval1 = np.zeros(M)
pval2 = np.zeros(M)
for i in range(M):
    pval1[i]= res[i][0]
    pval2[i]= res[i][1]
np.save('pval_sphere_var_debias' +'_tau_'+str(s1)+'_T_'+str(s2)+'.npy', pval1)
np.save('pval_sphere_var_bias' + '_tau_'+str(s1)+'_T_'+str(s2)+'.npy', pval2)
end1=time.time()
#print(end1-start1)
print(np.mean(pval1<=0.05))
print(np.mean(pval2<=0.05))
p.terminate()
elapsed_time = end1 - start1
formatted_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(elapsed_time))

print(f"Elapsed time: {formatted_time}")

