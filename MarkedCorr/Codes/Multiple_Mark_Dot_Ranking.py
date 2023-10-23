###### #code to initialise an optimiser given start points
######

import sys
print (sys.argv)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import os
from time import time

# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time 
from Pk_tools import Fourier, smooth_field, get_Pk
from numpy import sin as sin
from numpy import cos as cos
from numpy import sin as sin
from numpy import cos as cos
from numpy import arcsin as asin
from numpy import arccos as acos
import pandas as pd

from numpy import sin as sin
from numpy import cos as cos

import pandas as pd
from math import comb

# length_scale = float(sys.argv[1])

fom_type = sys.argv[1]  #type of figure of merit, both, om or s8
R = float(sys.argv[2]) #smoothing scale in Mpc
dot_threshold = 0.9 #this is out upper limit on dot product for checking similarity of curves, i.e. only dot product <0.9 curves are allowed when choosing next mark. 

print(f'fom type: {fom_type}')
print(f'R: {R}')


myvars=globals() 

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
    print('loading in', name)
    fields[f'{name}'] = np.load(f'/mnt/extraspace/jesscowell/MarkedCorr/Data/Sim_arrays/{name}_{Nmesh}_arr.npy')
    ktot, k_fields[f'{name}']= Fourier(fields[f'{name}'], Nmesh=Nmesh)
    myvars[f'fft_d_{name}'] = k_fields[f'{name}']

    smoothed_kfields[f'{name}'] = smooth_field(ktot, k_fields[f'{name}'], R, ) #IN K SPACE

    _, smoothed_fields[f'{name}'] = Fourier(smoothed_kfields[f'{name}'], inverse=True)
    delta_R[f'{name}']=smoothed_fields[f'{name}']-1
    k, nmodes_pk, pk = get_Pk(k_fields[f'{name}'], ktot)
    good_k =  k < kmax
    k = k[good_k]
    Pks[f'{name}'] = pk[good_k][3:]
    nmodes_pk = nmodes_pk[good_k][3:] #scale cuts
    Pks[f'{name}'] = pk

#scale cuts
Pk_dd_fid = Pks['fiducial'][good_k][3:]
om_finite_diff =(Pks['Om_p'][good_k][3:]- Pks['Om_m'][good_k][3:])/(2*delta_Om)
s8_finite_diff =(Pks['s8_p'][good_k][3:]- Pks['s8_m'][good_k][3:])/(2*delta_s8)

cov_dd_dd = np.diag((2*Pk_dd_fid**2)/nmodes_pk)
delta_R_fid = smoothed_fields['fiducial']-1


def my_pinv(cov, w_thr=1E-7):
    w, v = np.linalg.eigh(cov) #NB: cants use jax here because of <= statement below
    badw = w <= w_thr
    w_inv = 1./w
    w_inv[badw] = 0.
    pinv = jnp.dot(v, np.dot(np.diag(w_inv), v.T))
    return pinv 

def FOM(angles_array, length_scale, fom_type):
    ngrid=256
    nmodes=4
    delta_R_fid = smoothed_fields['fiducial']-1
    delta_R_train = jnp.linspace(np.min(delta_R_fid), 2.0, n_modes).reshape(-1,1) 
    kernel = 20*ConstantKernel(constant_value=1., constant_value_bounds=(0, 30.0))*RBF(length_scale=length_scale,) #this could be changed, no reason for this choice?
    
    mark_names=[] #will be array for storing names of fields
    for i in range(1,len(angles_array)+1): #4
        a,b,c = angles_array[i-1]
        w = sin(a)*sin(b)*sin(c)
        x = sin(a)*sin(b)*cos(c)
        y =sin(a)*cos(b)
        z = cos(a)
        myvars= globals() ### this var stuff is just a hack Im using for naming variables inside a loop easily
        mark_nodes= jnp.array([w,x,y,z,])
        myvars[f'mark_nodes{i}'] = jnp.array([w,x,y,z,])
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=1)
        gpr.fit(delta_R_train, mark_nodes) #this is where the gp fits a smooth mark curve!
        mark_names.append(f'm{i}')


    #calculate the marks as predicided from gpr for each field (will have a different array of delta_R for each)
        mark_fid,_ = gpr.predict((delta_R_fid.flatten()).reshape(-1,1), return_std=True)
        mark_Omp, _ = gpr.predict((delta_R['Om_p'].flatten()).reshape(-1,1), return_std=True)
        mark_Omm, _ = gpr.predict((delta_R['Om_m'].flatten()).reshape(-1,1), return_std=True)
        mark_s8p, _ = gpr.predict((delta_R['s8_p'].flatten()).reshape(-1,1), return_std=True)
        mark_s8m, _ = gpr.predict((delta_R['s8_m'].flatten()).reshape(-1,1), return_std=True)

        #then reshape back into 3D arrays
        myvars[f'mark_fid_{i}'] = mark_fid.reshape([ngrid, ngrid, ngrid])
        myvars[f'mark_Omp{i}'] = mark_Omp.reshape([ngrid, ngrid, ngrid])
        myvars[f'mark_Omm{i}']= mark_Omm.reshape([ngrid, ngrid, ngrid])
        myvars[f'mark_s8_p{i}'] = mark_s8p.reshape([ngrid, ngrid, ngrid])
        myvars[f'mark_s8_m{i}']= mark_s8m.reshape([ngrid, ngrid, ngrid])


        ################################
        #calculate the marked field in real space 

        marked_field_fid = fields[f'fiducial']*myvars[f'mark_fid_{i}']
        marked_field_omp = fields[f'Om_p']*myvars[f'mark_Omp{i}']
        marked_field_omm = fields[f'Om_m']*myvars[f'mark_Omm{i}']
        marked_field_s8p =fields[f's8_p']*myvars[f'mark_s8_p{i}']
        marked_field_s8m = fields[f's8_m']*myvars[f'mark_s8_m{i}']

        #fft marked field
        _, myvars[f'fft_m{i}_fiducial'] = Fourier(marked_field_fid, Nmesh=Nmesh)
        _, myvars[f'fft_m{i}_Om_p'] = Fourier(marked_field_omp, Nmesh=Nmesh)
        _, myvars[f'fft_m{i}_Om_m'] = Fourier(marked_field_omm, Nmesh=Nmesh)
        _, myvars[f'fft_m{i}_s8_p'] = Fourier(marked_field_s8p, Nmesh=Nmesh)
        _, myvars[f'fft_m{i}_s8_m'] = Fourier(marked_field_s8m, Nmesh=Nmesh)




    #############################################################

    


    def finite_diff_pair():
        ipk = 0
        map_names =  ['d']+ mark_names#diff Pk combos
        names = ['Om', 's8']

    #     names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p']
        for i1, n1 in enumerate(map_names):
    #         print('map names',map_names[i1:])
            for n2 in map_names[i1:]: #
    #             print(ipk, n1, n2)
                for n3 in names:
                    yield ipk, n1, n2, n3
                    ipk += 1
    #get power spectra
    def pairs():
        ipk = 0
        map_names = ['d']+ mark_names#diff Pk combos
        names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p',]

    #     names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p']
        for i1, n1 in enumerate(map_names):
    #         print('map names',map_names[i1:])
            for n2 in map_names: #[i1:]
    #             print(ipk, n1, n2)
                for n3 in names:
                    yield ipk, n1, n2, n3
                    ipk += 1
    def iterate_pairs():
        map_names = ['d'] + mark_names
        ipk = 0
        for i1, n1 in enumerate(map_names):
            for n2 in map_names[i1:]: 
                yield ipk, n1, n2
                ipk += 1


    for ix, f1, f2, f3 in pairs(): #diff fields m1dm, d2d etc, 
            _, _, myvars[f'Pk_{f1}{f2}_{f3}'] =  get_Pk(myvars[f'fft_{f1}_{f3}'], ktot, second =myvars[f'fft_{f2}_{f3}'])
            myvars[f'Pk_{f1}{f2}_{f3}'] =myvars[f'Pk_{f1}{f2}_{f3}'][good_k]
            myvars[f'Pk_{f1}{f2}_{f3}'] =myvars[f'Pk_{f1}{f2}_{f3}'][3:]
#     print(mark_names)
#     deriv_s8=deriv_Om=[]
    myvars['deriv_s8']= myvars['deriv_Om']=[]

    for ix, f1, f2, obs in finite_diff_pair():
        # print(f'{obs}_{f1}{f2}_finite_diff = ', f'Pk_{f1}{f2}_{obs}_p -Pk_{f1}{f2}_{obs}_m') 
        myvars[f'{obs}_{f1}{f2}_finite_diff'] = (myvars[f'Pk_{f1}{f2}_{obs}_p'] - myvars[f'Pk_{f1}{f2}_{obs}_m'])/(2*delta_Om)
        myvars[f'deriv_{obs}'] = (np.hstack([ myvars[f'deriv_{obs}'],myvars[f'{obs}_{f1}{f2}_finite_diff'] ]))

    length = comb(len(mark_names)+2,2)
    # print('length', length)
    ndata = len(deriv_s8)
    # print('ndata', ndata)
    idx=[]
    
    i=0 
    indices={}
    for ix, n1, n2, in iterate_pairs():
#         print(ix, n1, n2, obs)
        idx.append(f'id_{n1}{n2}' )
        myvars[f'id_{n1}{n2}'] =  np.arange(ndata).reshape([length, ndata//length])[i]
        indices[f'{n1}{n2}']= myvars[f'id_{n1}{n2}']
        # print(f'{n1}{n2}')
        i+=1

                    
    cov = np.zeros([ndata, ndata])
    
        
        
    for ipka, n1_a, n2_a in iterate_pairs():
        id_a = indices[f'{n1_a}{n2_a}']
        for ipkb, n1_b, n2_b in iterate_pairs():
            id_b = indices[f'{n1_b}{n2_b}']
            pk_n1a_n1b = myvars[f'Pk_{n1_a}{n1_b}_fiducial']
            pk_n2a_n2b = myvars[f'Pk_{n2_a}{n2_b}_fiducial']
            pk_n1a_n2b = myvars[f'Pk_{n1_a}{n2_b}_fiducial']
            pk_n2a_n1b = myvars[f'Pk_{n2_a}{n1_b}_fiducial']

            cov[np.ix_(id_a, id_b)] = np.diag((pk_n1a_n1b*pk_n2a_n2b +
                                               pk_n1a_n2b*pk_n2a_n1b)/nmodes_pk)


    
    icov = my_pinv(cov) #inverse covariance
    
#     fisher_dd = np.array([[jnp.dot(s8_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), s8_finite_diff)),
#                        jnp.dot(s8_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), om_finite_diff))],
#                       [jnp.dot(om_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), s8_finite_diff)),
#                        jnp.dot(om_finite_diff, jnp.dot(np.linalg.inv(cov_dd_dd), om_finite_diff))]])

    fisher_cov_so = jnp.dot(deriv_s8.T,np.dot( icov, deriv_Om))
    fisher_cov_os = jnp.dot(deriv_Om.T,np.dot( icov, deriv_s8))
    fisher_cov_ss = jnp.dot(deriv_s8.T,np.dot( icov, deriv_s8))
    fisher_cov_oo = jnp.dot(deriv_Om.T,np.dot( icov, deriv_Om))

    fisher_cov = np.array([[fisher_cov_ss, fisher_cov_so],
                           [fisher_cov_os, fisher_cov_oo]])
    

    #sanity checks 

    if np.isnan(fisher_cov_os):
        print(f'NAN VALUE for b={b}, p={p}, os')

    if np.isnan(fisher_cov_so):
        print(f'NAN VALUE for b={b}, p={p}, so')

    error = np.sqrt(my_pinv(fisher_cov))
    # print(fisher_cov,  'fisher_cov')

    cov_mark = np.linalg.inv(fisher_cov)
#     r_om_s8_mark = cov_mark[0, 1]/np.sqrt(cov_mark[0, 0]*cov_mark[1, 1])
#     cov_dd = np.linalg.inv(fisher_dd)
#     r_om_s8_dd = cov_dd[0, 1]/np.sqrt(cov_dd[0, 0]*cov_dd[1, 1])

    # print(scipy.linalg.issymmetric(cov))
    w, v = np.linalg.eigh(cov) #cants use jax here because of = statement below)
    # plt.show()
    print(fom_type)
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

#     print('error', error)
#     print('fom', fom)
    return(fom, mark_fid, fisher_cov )

def pos_def(mat):
    eigenvalues = np.linalg.eigvals(mat)
    if all(eigenvalues > 0):
        return("cov is pos def")
    else:
        return("cov is NOT pos def")
    


def retrieve_curve(dat, row):
    a = np.array(dat['a'])[row]
    b =np.array(dat['b'])[row]
    c =np.array(dat['c'])[row]
    return[a,b,c]


def retrieve_vec(dat, row):
    a = np.array(dat['a'])[row]
    b =np.array(dat['b'])[row]
    c =np.array(dat['c'])[row]
    w = sin(a)*sin(b)*sin(c)
    x = sin(a)*sin(b)*cos(c)
    y =sin(a)*cos(b)
    z = cos(a)
    return[w,x,y,z]
# icov_dd = my_pinv(cov_dd_dd)
# delta_R_fid = smoothed_fields['fiducial']-1


####################################################
#main code starts here
n_modes=4
print(R)
 

# R_arr=[ 30.0, 5.0, 10.0]

# fom_types=['om'] #, 's8' 'both']
# print('FOM TYPES')
# for R in R_arr:
    
#     for fom_type in fom_types:
    #R=10 case
#     if fom_type=='s8':
#         dat=pd.read_csv('S8_OPTIMISERS_RESULT')
#     elif fom_type=='om':
#         dat=pd.read_csv('optimiser_varying_start/OM_OPTIMISERS_RESULT')
#     elif fom_type=='both':
#         dat=pd.read_csv('optimiser_varying_start/BOTH_OPTIMISERS_RESULT')
        
dat=pd.read_csv(f'/mnt/zfsusers/jesscowell/MarkedCorr/Codes/Optimisers/raw_result_optimisers_fom_{fom_type}_R_{R}')
print(f'Length is {len(dat)}, now calculating {fom_type}, R={R}')
if len(dat)<100:
    raise exception('not enought optimisers')
     

myvars= globals() ### this var stuff is just a hack for naming variables inside a loop
# fom1 = dat['FOM']
# np.save('fom1_s8',fom1 )
# folder='/mnt/extraspace/jesscowell/MarkedCorr/PINV_DOT_TESTS/'
folder=f'/mnt/extraspace/jesscowell/MarkedCorr/PINV_DOT_{dot_threshold}/'
for i in range (1,15): #changed from 15 to 2 for plots
    print(f'{fom_type}, R={R} progress{i}/20')
    if i == 1:
        print('starting')
        dat=pd.read_csv(f'/mnt/zfsusers/jesscowell/MarkedCorr/Codes/Optimisers/raw_result_optimisers_fom_{fom_type}_R_{R}')
#         myvars[f'fom{i}'] = pd.read_csv(folder+f'fom{i}_{fom_type}_R={R}')
#         myvars[f'idx{i}'] = np.argmax(myvars[f'fom{i}']['fom'])
#         myvars[f'curve{i}'] = retrieve_curve(dat, myvars[f'idx{i}'])      
#         myvars[f'curve{i}_vec'] = retrieve_vec(dat, myvars[f'idx{i}'])      
   
#         dot_arr=[]
#         for row in range(len(dat)):
#             curve_vec = np.array(retrieve_vec(dat, row))
#             dot = abs(np.array(myvars[f'curve{i}_vec'])@curve_vec)
#             print(dot)
            
#             dot_arr.append(dot) 
#         good_dots = np.array(dot_arr)  <= 0.9
#         print('good dots', good_dots)
#         next_dat = dat[good_dots]
#         print(len(next_dat), 'length!')
#         next_dat.to_csv(folder+f'dat{i+1}_{fom_type}_R={R}')
    else:
        dat = pd.read_csv((folder+f'dat{i}_{fom_type}_R={R}'))
    if os.path.isfile(folder+f'fom{i}_{fom_type}_R={R}'):
        myvars[f'fom{i}'] = pd.read_csv(folder+f'fom{i}_{fom_type}_R={R}')
        myvars[f'idx{i}'] = np.argmax(myvars[f'fom{i}']['fom'])
        myvars[f'curve{i}'] = np.array(retrieve_curve(dat, myvars[f'idx{i}'])    )  
        myvars[f'curve{i}_vec'] = retrieve_vec(dat, myvars[f'idx{i}'])      
        
    else:
        print('not found',folder+f'fom{i}_{fom_type}_R={R}')
        cols=['fom', 'type', 'fisher_matrix00','fisher_matrix01','fisher_matrix10','fisher_matrix11', 'max_indices']
#         myvars[f'fom{i}']=[]

        myvars[f'fom{i}']= pd.DataFrame(columns = cols)

          

        print('length of dat is', len(dat))
        for row in range(len(dat)):
            df = pd.DataFrame(columns = cols, data=np.zeros((1, len(cols)) ))
            curve_vec = retrieve_vec(dat, row)
            curve = retrieve_curve(dat, row)
            curves = [myvars[f'curve{ix}'] for ix in range(1,i)]
            curves.append(curve)
            fom, mark_curve, fish_mat = FOM(curves, 0.5, fom_type=fom_type)
            df['fom'] =fom
            df['fisher_matrix00'] =fish_mat[0,0]
            df['fisher_matrix01'] =fish_mat[0,1]
            df['fisher_matrix10'] =fish_mat[1,0]
            df['fisher_matrix11'] =fish_mat[1,1]
            df['type'] = fom_type
#             df['curve'] = mark_curve
#             df['delta_fid'] = delta_R
#                     df['max_indices'] = np.argmax(myvars[f'fom{i}'])
            myvars[f'fom{i}'] = pd.concat([myvars[f'fom{i}'], df])
         
          
        myvars[f'idx{i}'] = np.argmax(myvars[f'fom{i}']['fom']) #max index
        myvars[f'curve{i}'] = retrieve_curve(dat, myvars[f'idx{i}'])   #max curve  
        myvars[f'curve{i}_vec'] = retrieve_vec(dat, myvars[f'idx{i}'])      
        
        dot_arr=[]
        for row in range(len(dat)):
            curve_vec = retrieve_vec(dat, row)
            
            dot_arr.append(abs(np.array(myvars[f'curve{i}_vec'])@np.array(curve_vec))) 
        good_dots = np.array(dot_arr)  <= dot_threshold
        next_dat = dat[good_dots]
        print(len(next_dat), 'length!')
        next_dat.to_csv(folder+f'dat{i+1}_{fom_type}_R={R}')
        
                         
        myvars[f'fom{i}'].to_csv(folder+f'fom{i}_{fom_type}_R={R}')
#                 np.save(f'fom{i}_{fom_type}.npy', myvars[f'fom{i}'])
        myvars[f'idx{i}'] = np.argmax(myvars[f'fom{i}']['fom'])
        myvars[f'curve{i}'] = retrieve_curve(dat, myvars[f'idx{i}'])         


        
        