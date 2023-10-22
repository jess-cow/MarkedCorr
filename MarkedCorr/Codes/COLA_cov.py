import nbodykit
import numpy as np
from nbodykit.lab import *
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib
import os
import time
import pandas as pd
from math import comb
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from Pk_tools import Fourier, smooth_field, get_Pk
# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from numpy import sin as sin
from numpy import cos as cos
ngrid=256
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
    
# loading in sims + calculating power spectra, this takes a lon g time so maybe use saved ones for now


#to do:
#load in the marks
#all the pk
#check shape of cov compares
fom_type='both'
R= 10.0
n_modes=4
folder='/mnt/extraspace/jesscowell/MarkedCorr/PINV_DOT_0.9/'
myvars=globals()
angles_array=[]
for i in range (1,14):
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
        angles_array.append(myvars[f'curve{i}'])
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
        good_dots = np.array(dot_arr)  <= 0.9
        next_dat = dat[good_dots]
        print(len(next_dat), 'length!')
        next_dat.to_csv(folder+f'dat{i+1}_{fom_type}_R={R}')
        
                         
        myvars[f'fom{i}'].to_csv(folder+f'fom{i}_{fom_type}_R={R}')
#                 np.save(f'fom{i}_{fom_type}.npy', myvars[f'fom{i}'])
        myvars[f'idx{i}'] = np.argmax(myvars[f'fom{i}']['fom'])
        myvars[f'curve{i}'] = retrieve_curve(dat, myvars[f'idx{i}'])         


        

    
print(curve2)

direc= '/mnt/extraspace/damonge/MarkedPk/fiducial_512/COLA/'
# angles_array=np.array([curve1, curve2, curve3, curve4])
Nmesh=256
kmin= 0.01
length_scale=0.75
n_modes=4
kmax = .3 #max k for scale cuts 
for seed in range(1000, 2000 ): ##2000
    start=time.time()
    
    catalog= Gadget1Catalog(direc+f"seed{seed}/sn_fiducial_512_seed0{seed}_001.0")
    
#     print(i)
#     catalog = Gadget1Catalog(direc+f"seed{seed}/")
    for i in range(1, 64):
        new = Gadget1Catalog(direc+f"seed{seed}/sn_fiducial_512_seed0{seed}_001.{i}")
        catalog= transform.ConcatenateSources(catalog, new)
    mesh = catalog.to_mesh(Nmesh=Nmesh)

    rfield = mesh.paint(mode='real') #get real field
    delta = np.array(rfield)
    delta_R_fid=delta-1
    name='fiducial'
    
    
    sims={}; mesh = {}; power={}; painted_arr={}; fields={}
    k_fields={}; smoothed_kfields={}; Pks={}; smoothed_fields={}
    delta_R={}
    
    fields['fiducial'] = delta
    
    
    ktot, k_fields[f'{name}']= Fourier(fields[f'{name}'], Nmesh=Nmesh)
    myvars[f'fft_d_{name}'] = k_fields[f'{name}']

    smoothed_kfields[f'{name}'] = smooth_field(ktot, k_fields[f'{name}'], R, ) #IN K SPACE

    _, smoothed_fields[f'{name}'] = Fourier(smoothed_kfields[f'{name}'], inverse=True)
    delta_R[f'{name}']=smoothed_fields[f'{name}']-1
    k, nmodes_pk, pk = get_Pk(k_fields[f'{name}'], ktot)
    good_k =  k < kmax
#     k = k[good_k]
#     Pks[f'{name}'] = pk[good_k][3:]
#     nmodes_pk = nmodes_pk[good_k][3:] #scale cuts
#     Pks[f'{name}'] = pk
    
    print(f'loaded catalog {seed}, took {time.time()-start} seconds')
    # def FOM(angles_array, length_scale, fom_type, delta_R, delta_R_fid):
#     '''Calculates FOM'''
#     ngrid=256
    nmodes=4
#     delta_R_fid = smoothed_fields['fiducial']-1
    delta_R_train = jnp.linspace(np.min(delta_R_fid), 2.0, n_modes).reshape(-1,1) 
    kernel = 20*ConstantKernel(constant_value=1., constant_value_bounds=(0, 30.0))*RBF(length_scale=length_scale,) #this could be changed     
    mark_names=[]
    for i in range(1,len(angles_array)+1): #4
        #doing calculations for each mark inside here

        a,b,c = angles_array[i-1]
        w = sin(a)*sin(b)*sin(c)
        x = sin(a)*sin(b)*cos(c)
        y =sin(a)*cos(b)
        z = cos(a)
        myvars= globals() ### this var stuff is just a hack for naming variables inside a loop
        mark_nodes= jnp.array([w,x,y,z,])
        myvars[f'mark_nodes{i}'] = jnp.array([w,x,y,z,])
        # vars[f'gpr{i}''] 
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=1)
        gpr.fit(delta_R_train, mark_nodes)
        mark_names.append(f'm{i}')


        #calculate the marks as predicided from gpr
        mark_fid,_ = gpr.predict((delta_R_fid.flatten()).reshape(-1,1), return_std=True)
#         mark_Omp, _ = gpr.predict((delta_R['Om_p'].flatten()).reshape(-1,1), return_std=True)
#         mark_Omm, _ = gpr.predict((delta_R['Om_m'].flatten()).reshape(-1,1), return_std=True)
#         mark_s8p, _ = gpr.predict((delta_R['s8_p'].flatten()).reshape(-1,1), return_std=True)
#         mark_s8m, _ = gpr.predict((delta_R['s8_m'].flatten()).reshape(-1,1), return_std=True)

        #then reshape back into 3D arrays
        myvars[f'mark_fid_{i}'] = mark_fid.reshape([ngrid, ngrid, ngrid])
#         myvars[f'mark_Omp{i}'] = mark_Omp.reshape([ngrid, ngrid, ngrid])
#         myvars[f'mark_Omm{i}']= mark_Omm.reshape([ngrid, ngrid, ngrid])
#         myvars[f'mark_s8_p{i}'] = mark_s8p.reshape([ngrid, ngrid, ngrid])
#         myvars[f'mark_s8_m{i}']= mark_s8m.reshape([ngrid, ngrid, ngrid])


        ################################
        #calculate the marked field in real space 

        marked_field_fid = fields[f'fiducial']*myvars[f'mark_fid_{i}']
#         marked_field_omp = fields[f'Om_p']*myvars[f'mark_Omp{i}']
#         marked_field_omm = fields[f'Om_m']*myvars[f'mark_Omm{i}']
#         marked_field_s8p =fields[f's8_p']*myvars[f'mark_s8_p{i}']
#         marked_field_s8m = fields[f's8_m']*myvars[f'mark_s8_m{i}']

        #fft marked field
        _, myvars[f'fft_m{i}_fiducial'] = Fourier(marked_field_fid, Nmesh=Nmesh)
#         _, myvars[f'fft_m{i}_Om_p'] = Fourier(marked_field_omp, Nmesh=Nmesh)
#         _, myvars[f'fft_m{i}_Om_m'] = Fourier(marked_field_omm, Nmesh=Nmesh)
#         _, myvars[f'fft_m{i}_s8_p'] = Fourier(marked_field_s8p, Nmesh=Nmesh)
#         _, myvars[f'fft_m{i}_s8_m'] = Fourier(marked_field_s8m, Nmesh=Nmesh)


    #should now have all marks saved as variables
    #############################################################
    map_names = ['d']+ mark_names#diff Pk combos
    
    

    #get power spectra
    def pairs():
        ipk = 0
        names = ['fiducial',]
        for i1, n1 in enumerate(map_names):
            for n2 in map_names: #[i1:]
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
    Pk_arr=[]
    

    for ix, f1, f2, f3 in pairs(): #diff fields m1dm, d2d etc,
        #get Pk!!!! should save this somewhere....
            _, _, myvars[f'Pk_{f1}{f2}_{f3}'] =  get_Pk(myvars[f'fft_{f1}_{f3}'], ktot, second =myvars[f'fft_{f2}_{f3}'])
            myvars[f'Pk_{f1}{f2}_{f3}'] =myvars[f'Pk_{f1}{f2}_{f3}'][good_k][3:] #scale cuts
            Pk_arr.append(myvars[f'Pk_{f1}{f2}_{f3}'])
        
    path = '/mnt/extraspace/jesscowell/MarkedCorr/COLA_PKs/'
    np.save(path+f'seed{seed}_Pks', Pk_arr)

    print(f'overall took {time.time()-start} seconds')