######
#code to initialise an optimiser given start points
######

import sys
print (sys.argv)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from time import time
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
# import nbodykit
import numpy as np
# from nbodykit.lab import *
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib
import os
import time 
from Pk_tools import Fourier, smooth_field, get_Pk
import scipy
from numpy import sin as sin
from numpy import cos as cos

from numpy import sin as sin
from numpy import cos as cos
from numpy import arcsin as asin
from numpy import arccos as acos
import pandas as pd
length_scale = float(sys.argv[1])
fom_type = sys.argv[2]
R = float(sys.argv[3])
number_points=float(sys.argv[4])
####################
#function defs
####################
def my_pinv(cov, w_thr=1E-3):
    w, v = np.linalg.eigh(cov) #cants use jax here because of = statement below
    badw = w <= w_thr
    w_inv = 1./w
    w_inv[badw] = 0.
    pinv = jnp.dot(v, np.dot(np.diag(w_inv), v.T))
    return pinv 


def pos_def(mat):
    eigenvalues = np.linalg.eigvals(mat)
    if all(eigenvalues > 0):
        return("cov is pos def")
    else:
        return("cov is NOT pos def")
    

def FOM(mark_angles, length_scale):
    a,b,c = mark_angles
    
    w = sin(a)*sin(b)*sin(c)
    x = sin(a)*sin(b)*cos(c)
    y =sin(a)*cos(b)
    z = cos(a)
    
    mark_nodes = jnp.array([w,x,y,z,])
    n_modes=4
    ngrid = 256
    print('mark nodes', mark_nodes)
    kernel = 20*ConstantKernel(constant_value=1., constant_value_bounds=(0, 30.0))*RBF(length_scale=length_scale,) #this could be changed

    ##############################
    #set up the GP
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=1)

    #############################
    #train 
    
    # 4 points in delta_R, should remain same each iteration in optimiser
    delta_R_train = jnp.linspace(np.min(delta_R_fid), 2.0, n_modes).reshape(-1,1) 
    gpr.fit(delta_R_train, mark_nodes)
    
    ###############################
    #predict mark for each field
    
    mark_fid, y_std = gpr.predict((delta_R_fid.flatten()).reshape(-1,1), return_std=True)
   
    # x = np.random.rand(500)* (np.max(delta_R_fid.flatten())-np.min(delta_R_fid.flatten()))+ np.min(delta_R_fid.flatten())

  

    mark_Omp, _ = gpr.predict((delta_R['Om_p'].flatten()).reshape(-1,1), return_std=True)
    mark_Omm, _ = gpr.predict((delta_R['Om_m'].flatten()).reshape(-1,1), return_std=True)
    mark_s8p, _ = gpr.predict((delta_R['s8_p'].flatten()).reshape(-1,1), return_std=True)
    mark_s8m, _ = gpr.predict((delta_R['s8_m'].flatten()).reshape(-1,1), return_std=True)
    
    #then reshape back into 3D arrays
    mark_fid =  mark_fid.reshape([ngrid, ngrid, ngrid])
    mark_Omp =  mark_Omp.reshape([ngrid, ngrid, ngrid])
    mark_Omm =  mark_Omm.reshape([ngrid, ngrid, ngrid])
    mark_s8p =  mark_s8p.reshape([ngrid, ngrid, ngrid])
    mark_s8m =  mark_s8m.reshape([ngrid, ngrid, ngrid])

    
    
    ################################
    #calculate the marked field in real space 
    
    #todo: change these to right marks
    marked_field_fid = fields[f'fiducial']*mark_fid
    marked_field_omp = fields[f'Om_p']*mark_Omp
    marked_field_omm = fields[f'Om_m']*mark_Omm
    marked_field_s8p =fields[f's8_p']*mark_s8p
    marked_field_s8m = fields[f's8_m']*mark_s8m

    #fft marked field
    _, fft_marked_fid = Fourier(marked_field_fid, Nmesh=Nmesh)
    _, fft_marked_omp = Fourier(marked_field_omp, Nmesh=Nmesh)
    _, fft_marked_omm = Fourier(marked_field_omm, Nmesh=Nmesh)
    _, fft_marked_s8p = Fourier(marked_field_s8p, Nmesh=Nmesh)
    _, fft_marked_s8m = Fourier(marked_field_s8m, Nmesh=Nmesh)
    # print('MARKED FFT FIELD___________________', fft_marked_fid)

    #get power spectra
    _, _, Pk_md_fid = get_Pk(fft_marked_fid, ktot, second = k_fields['fiducial'], )
    _, _, Pk_mm_fid = get_Pk(fft_marked_fid,ktot)
    _, _, Pk_md_Omp = get_Pk(fft_marked_omp, ktot, second = k_fields['Om_p'], )
    _, _, Pk_mm_Omp = get_Pk(fft_marked_omp,ktot)
    _, _, Pk_md_Omm = get_Pk(fft_marked_omm, ktot, second = k_fields['Om_m'], )
    _, _, Pk_mm_Omm = get_Pk(fft_marked_omm,ktot)
    _, _, Pk_md_s8p = get_Pk(fft_marked_s8p, ktot, second = k_fields['s8_p'])
    _, _, Pk_mm_s8p = get_Pk(fft_marked_s8p, ktot)
    _, _, Pk_md_s8m = get_Pk(fft_marked_s8m, ktot, second = k_fields['s8_m'])
    _, _, Pk_mm_s8m = get_Pk(fft_marked_s8m, ktot)

    #scale cuts --to doL check if need this

    Pk_dd_fid = Pks['fiducial'][good_k][3:]
    Pk_md_fid = Pk_md_fid[good_k]
    Pk_md_s8m = Pk_md_s8m[good_k]
    Pk_md_s8p = Pk_md_s8p[good_k]
    Pk_md_Omm = Pk_md_Omm[good_k]
    Pk_md_Omp = Pk_md_Omp[good_k]
    Pk_mm_fid = Pk_mm_fid[good_k]
    Pk_mm_s8m = Pk_mm_s8m[good_k]
    Pk_mm_s8p = Pk_mm_s8p[good_k]
    Pk_mm_Omm = Pk_mm_Omm[good_k]
    Pk_mm_Omp = Pk_mm_Omp[good_k]

    Pk_md_fid = Pk_md_fid[3:]
    Pk_md_s8m = Pk_md_s8m[3:]
    Pk_md_s8p = Pk_md_s8p[3:]
    Pk_md_Omm = Pk_md_Omm[3:]
    Pk_md_Omp = Pk_md_Omp[3:]
    Pk_mm_fid = Pk_mm_fid[3:]
    Pk_mm_s8m = Pk_mm_s8m[3:]
    Pk_mm_s8p = Pk_mm_s8p[3:]
    Pk_mm_Omm = Pk_mm_Omm[3:]
    Pk_mm_Omp = Pk_mm_Omp[3:]


    #calculate finite difference
    om_marked_finite_diff =(Pk_mm_Omp - Pk_mm_Omm)/(2*delta_Om)
    s8_marked_finite_diff =(Pk_mm_s8p - Pk_mm_s8m)/(2*delta_s8)

    om_cross_finite_diff =(Pk_md_Omp - Pk_md_Omm)/(2*delta_Om)
    s8_cross_finite_diff =(Pk_md_s8p - Pk_md_s8m)/(2*delta_s8)

    #make vector of derivatives
    deriv_s = np.hstack([s8_finite_diff, s8_cross_finite_diff, s8_marked_finite_diff])
    deriv_o = np.hstack([om_finite_diff, om_cross_finite_diff, om_marked_finite_diff])

    # TODO: check the formulas below are correct
    ndata = len(deriv_s)

    cov = np.zeros([ndata, ndata])
    
    cov_dd_dd = np.diag(2*Pk_dd_fid**2/nmodes_pk)
    cov_dd_md = np.diag(2*Pk_dd_fid*Pk_md_fid/nmodes_pk)
    cov_dd_mm = np.diag(2*Pk_md_fid**2/nmodes_pk)
    cov_md_md = np.diag((Pk_dd_fid*Pk_mm_fid+Pk_md_fid**2)/nmodes_pk)
    cov_md_mm = np.diag(2*Pk_mm_fid*Pk_md_fid/nmodes_pk)
    cov_mm_mm = np.diag(2*Pk_mm_fid**2/nmodes_pk)
    id_dd, id_md, id_mm = np.arange(ndata).reshape([3, ndata//3])
    cov[np.ix_(id_dd, id_dd)] = cov_dd_dd
    cov[np.ix_(id_dd, id_md)] = cov_dd_md
    cov[np.ix_(id_dd, id_mm)] = cov_dd_mm
    cov[np.ix_(id_md, id_md)] = cov_md_md
    cov[np.ix_(id_md, id_mm)] = cov_md_mm
    cov[np.ix_(id_mm, id_mm)] = cov_mm_mm
    cov[np.ix_(id_md, id_dd)] = cov_dd_md.T
    cov[np.ix_(id_mm, id_dd)] = cov_dd_mm.T
    cov[np.ix_(id_mm, id_md)] = cov_md_mm.T
    icov = my_pinv(cov)

    # calculate fisher 
    fisher_dd = np.array([[jnp.dot(s8_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), s8_finite_diff)),
                           jnp.dot(s8_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), om_finite_diff))],
                          [jnp.dot(om_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), s8_finite_diff)),
                           jnp.dot(om_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), om_finite_diff))]])
    fisher_cov_so = jnp.dot(deriv_s.T,np.dot( icov, deriv_o))
    fisher_cov_os = jnp.dot(deriv_o.T,np.dot( icov, deriv_s))
    fisher_cov_ss = jnp.dot(deriv_s.T,np.dot( icov, deriv_s))
    fisher_cov_oo = jnp.dot(deriv_o.T,np.dot( icov, deriv_o))

    fisher_cov = np.array([[fisher_cov_ss, fisher_cov_so],[fisher_cov_os, fisher_cov_oo]])

    #sanity checks 

    if np.isnan(fisher_cov_os):
        print(f'NAN VALUE for b={b}, p={p}, os')

    if np.isnan(fisher_cov_so):
        print(f'NAN VALUE for b={b}, p={p}, so')
        
    error = np.sqrt(my_pinv(fisher_cov))
    

    cov_mark = np.linalg.inv(fisher_cov)
    r_om_s8_mark = cov_mark[0, 1]/np.sqrt(cov_mark[0, 0]*cov_mark[1, 1])
    cov_dd = np.linalg.inv(fisher_dd)
    r_om_s8_dd = cov_dd[0, 1]/np.sqrt(cov_dd[0, 0]*cov_dd[1, 1])
    
    if fom_type =='both':

#     print('error matrix', np.diag(error))
        fom = np.linalg.det(fisher_cov)
    elif fom_type =='s8':
        fom = 1/error[0,0]

    elif fom_type =='om':
        fom = 1/error[1,1]
    
    else:
        print('invalid FOM type!!!!')
        raise Exception

    error_dd= np.sqrt(np.diag(my_pinv(fisher_dd)))
#     print('fisher error', np.sqrt(np.diag(my_pinv(fisher_dd))))
    print('######error improvement######', np.diag(error)/error_dd*100)


    print('-fom', -fom)
    return -fom


def convert_to_angle(points):
    w, x, y, z = points
    rad = np.sqrt(x**2+y**2+z**2+w**2)
    
#     print('rad', rad)
    x /= rad
    y /= rad
    z /= rad
    w /= rad
  

    a = acos(z) #should be certain, range for a is 0,pi
    
    if sin(a) == 0:
        b, c = np.pi 
        return(a, b, c)
    
    
    b = acos(y/sin(a)) # range for b is also 0, pi 
    cosb = y/sin(a)

  
    if sin(b)==0:
        c= np.pi
        return(a, b, c)
    
   
    c = acos(x/(sin(a)*sin(b)))
    sinc = w/(sin(a)*sin(b))
    cosc = x/(sin(a)*sin(b))
    
    if sinc<0:
        c = 2*np.pi - c
        
        
   
    c2 = asin(w/(sin(a)*sin(b)))
    if cosc < 0: 
            c2 = np.pi - c2
        
    elif c2 < 0:
        c2 +=2*np.pi

    w = sin(a)*sin(b)*sin(c)
    x = sin(a)*sin(b)*cos(c)
    y =sin(a)*cos(b)
    z = cos(a)
    
    return(a,b,c)


# print(convert_to_angle(points)-points)
# def convert_to_angle(points):
#     w, x, y, z = points
#     rad = np.sqrt(x**2+y**2+z**2+w**2)
#     # print('rad', rad)
#     x /= rad
#     y /= rad
#     z /= rad
#     w /= rad
  

#     a = acos(z)
#     if sin(a) == 0:
#         b, c = np.pi 
#         return(a, b, c)
    
#     b = acos(y/sin(a))
#     if sin(b)==0:
#         c= np.pi
#         return(a, b, c)
    
#     c = acos(x/(sin(a)*sin(b)))
#     if sin(c) == 0:
#         d = np.pi
#         return(a, b, c, d)   
    
#     # print('mult', y/(sin(a)*sin(b)*sin(c)))
#     if  abs(y/(sin(a)*sin(b)*sin(c))) > 1.0:
#         d = asin(1.0)   
#         print('hacky fix',y/(sin(a)*sin(b)*sin(c)))
#         return(a, b, c, d)   
#     d = asin(y/(sin(a)*sin(b)*sin(c)))   
#     # print('d is', d)
#     return(a, b, c)


    
#######################################
#     '''load in sims'''

#information for loading in sims
names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p',]
seed = '101'; snapnum = '005'

Nmesh=256 #resolution
kmin= 0.01
kmax = .3 #max k for scale cuts 
delta_Om =  delta_s8 = 0.02

sims={}; mesh = {}; power={}; painted_arr={}; fields={}
k_fields={}; smoothed_kfields={}; Pks={}; smoothed_fields={}
delta_R={}
for name in names:
    print(name)
    fields[f'{name}'] = np.load(f'/mnt/zfsusers/jesscowell/MarkedCorr/Data/Sim_arrays/{name}_{Nmesh}_arr.npy')
    ktot, k_fields[f'{name}']= Fourier(fields[f'{name}'], Nmesh=Nmesh)

    smoothed_kfields[f'{name}'] = smooth_field(ktot, k_fields[f'{name}'], R, ) #IN K SPACE

    _, smoothed_fields[f'{name}'] = Fourier(smoothed_kfields[f'{name}'], inverse=True)
    delta_R[f'{name}']=smoothed_fields[f'{name}']-1
    k, nmodes_pk, pk = get_Pk(k_fields[f'{name}'], ktot)
    good_k =  k < kmax
    k = k[good_k]
    Pks[f'{name}'] = pk[good_k][3:]
    nmodes_pk = nmodes_pk[good_k]
    nmodes_pk = nmodes_pk[3:]

#     k = k[2:]
    Pks[f'{name}'] = pk

Pk_dd_fid = Pks['fiducial'][good_k][3:]

om_finite_diff =(Pks['Om_p'][good_k][3:]- Pks['Om_m'][good_k][3:])/(2*delta_Om)
s8_finite_diff =(Pks['s8_p'][good_k][3:]- Pks['s8_m'][good_k][3:])/(2*delta_s8)

cov_dd_dd = np.diag((2*Pk_dd_fid**2)/nmodes_pk)
delta_R_fid = smoothed_fields['fiducial']-1

icov_dd = my_pinv(cov_dd_dd)
##################################################################################
#randomly generate intial points

# Generate an n-dimensional vector of normal deviates (it suffices to use N(0, 1), 
x1, x2, x3, x4 = np.random.normal(0, 1, 4)
r = np.sqrt(x1**2 + x2**2 + x3**2 + x4**2)
print(x1/r)


initial_guess= convert_to_angle((x1,x2,x3,x4))



bound=(0,np.pi)
bound2=(0,2*np.pi)
start=time.time()
columns = ['']
cols = ['initial_guess', 'a,', 'b', 'c','FOM', '#iteration', 'FOM_type']

for i in range(0,100 ):
    #randomly generate intial points
    # Generate an n-dimensional vector of normal deviates (it suffices to use N(0, 1), 
    x1, x2, x3, x4 = np.random.normal(0, 1, 4)
    r = np.sqrt(x1**2 + x2**2 + x3**2 + x4**2)
    points= np.array([x1,x2,x3,x4])/r
    initial_guess= convert_to_angle(points)
    df = pd.DataFrame(columns = cols, data = np.zeros((1,len(cols))))
    df['initial_guess'] = [points]
    res = scipy.optimize.minimize(FOM, initial_guess, args=(length_scale), method='powell', bounds=[bound,bound,bound2])
    df['a'], df['b'], df['c'] = res.x
    df['FOM'] = res.fun
    df['#iteration'] = res.nit
    df['FOM_type']=  fom_type
    df.to_csv(f'optimiser_varying_start/{fom_type}_{i}_{number_points}points_R_{R}')


print('time taken', time.time()-start)
print('OPTIMAL RESULT', res)



#########
#saving output
# save_str = f'nelder_{fom_type}_length_{length_scale}_smooth_{R}Mpc.txt'
# np.savetxt(save_str, res.x)
# np.savetxt('itt_'+save_str, res.ni )