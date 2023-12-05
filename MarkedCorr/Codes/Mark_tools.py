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
import jax
import jax.numpy as jnp
# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)

from numpy import sin as sin
from numpy import cos as cos
from numpy import arcsin as asin
from numpy import arccos as acos
ngrid=256



#####
#check!!
######

def finite_diff_pair(mark_names,):
    ''''ix, f1, f2, obs'''
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
def pairs(mark_names, names):
    '''Generates all possible pairings for calculating Pks. Give mark names e.g. (m1, m2, m3), and names for all fields used, e.g. names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p']
    NB when doing finite diff need Om_m etc, bit for general covariance just need fiducial.
     {mark},{mark},{field} '''
    ipk = 0
    map_names = ['d']+ mark_names#diff Pk combos
    # names = ['fiducial',] 
#     names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p']
    for i1, n1 in enumerate(map_names):
#         print('map names',map_names[i1:])
        for n2 in map_names: #[i1:]
#             print(ipk, n1, n2)
            for n3 in names:
                yield ipk, n1, n2, n3
                ipk += 1


def iterate_pairs(mark_names):
    '''
    index and pairing d x m1 etc, 
    '''
    map_names = ['d'] + mark_names
    ipk = 0
    for i1, n1 in enumerate(map_names):
        for n2 in map_names[i1:]: 
            yield ipk, n1, n2
            ipk += 1


def my_pinv(cov, w_thr=1E-7):
    '''Calculate pseudo inverse of a covariance matrix.
    Parameters:
    cov: covariance matrix
    w_thr: threshold of lowest acceptable eigenvalue RATIO, n.b. we updated this '''
    w, v = np.linalg.eigh(cov) #cants use jax here because of = statement below
#     cond_n = np.max(w)/w
    inv_cond = w/np.max(w)
    badw = inv_cond < w_thr
    w_inv = 1./w
    w_inv[badw] = 0.
    print('number cut', np.sum(badw))
    print('badw', badw)

    
    print('max condition number',np.max(1/inv_cond))
#     print(‘final condition number’,np.max(1/inv_cond[~badw]))
    pinv = jnp.dot(v, np.dot(np.diag(w_inv), v.T))
    return pinv
def retrieve_curve(dat, row):
    '''Retrieve 3 angles corresponding to a mark function, just for the .dat files I created to neaten code up.
    Parameters:
    dat: table of mark functions
    row: index of the mark you want'''
    a = np.array(dat['a'])[row]
    b =np.array(dat['b'])[row]
    c =np.array(dat['c'])[row]
    return[a,b,c]


def retrieve_vec(dat, row):
    '''Same as retireve curve, but with coord transform to f(delta_R) space. Retrieve 4 points corresponding to a mark function, just for the .dat files I created to neaten code up.
    Parameters:
    dat: table of mark functions
    row: index of the mark you want'''
    a = np.array(dat['a'])[row]
    b =np.array(dat['b'])[row]
    c =np.array(dat['c'])[row]
    w = sin(a)*sin(b)*sin(c)
    x = sin(a)*sin(b)*cos(c)
    y =sin(a)*cos(b)
    z = cos(a)
    return[w,x,y,z]


def load_sims(R=10):
    '''
    Load in simulation data, given a smoothing scale in Mpc
    Returns:
    fields: array of overdensity fields
    Pks: dictionary of Pks for all fields
    smoothed_fields: dictionary of smoothed overdensity fields
    nmodes_pk: Array specifying number of k-modes in each Pk
    
    '''
    
    names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p',]
    seed = '101'; snapnum = '005'

    Nmesh=256 #resolution
    kmax = .3 #max k for scale cuts 
    delta_Om =  delta_s8 = 0.02

    sims={}; mesh = {}; power={}; painted_arr={}; fields={}; rho_fields={}
    k_fields={}; smoothed_kfields={}; Pks={}; smoothed_fields={}
    delta_R={}
    for name in names:
        print(name)
        rho_fields[f'{name}'] = np.load(f'/mnt/extraspace/jesscowell/MarkedCorr/Data/Sim_arrays/{name}_{Nmesh}_arr.npy')
#         print(np.mean(rho_fields[f'{name}']))

        fields[f'{name}'] =  rho_fields[f'{name}']/np.mean(rho_fields[f'{name}']) -1
#         print(np.mean(fields[f'{name}']))


#         ktot, k_fields[f'{name}']= Fourier(rho_fields[f'{name}'], Nmesh=Nmesh)
#         smoothed_kfields[f'{name}'] = smooth_field(ktot, k_fields[f'{name}'], R, ) #IN K SPACE
#         _, smoothed_fields[f'{name}'] = Fourier(smoothed_kfields[f'{name}'], inverse=True)
# #         print/('og', np.mean(smoothed_fields[f'{name}']-1))

        ktot, k_fields[f'{name}']= Fourier(fields[f'{name}'], Nmesh=Nmesh)
        smoothed_kfields[f'{name}'] = smooth_field(ktot, k_fields[f'{name}'], R, ) #IN K SPACE
        _, smoothed_fields[f'{name}'] = Fourier(smoothed_kfields[f'{name}'], inverse=True)

#         print('correct',np.mean(smoothed_fields[f'{name}']))

        delta_R[f'{name}']=smoothed_fields[f'{name}']-1
        k, nmodes_pk, pk = get_Pk(k_fields[f'{name}'], ktot)
        good_k =  k < kmax #for scale cuts
        k = k[good_k][3:] #throw away first 3 points also
        Pks[f'{name}'] = pk[good_k][3:]
        nmodes_pk = nmodes_pk[good_k][3:]
    

    return(ktot,good_k, Pks, fields, k_fields,smoothed_fields, nmodes_pk)

def convert_to_angle(points):
    '''convert from delta_R space to angles, i.e. cartesian 4D on surface of sphere to angles'''
    w, x, y, z = points
    rad = np.sqrt(x**2+y**2+z**2+w**2) #radius, normalisation
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
        

    w = sin(a)*sin(b)*sin(c)
    x = sin(a)*sin(b)*cos(c)
    y =sin(a)*cos(b)
    z = cos(a)
    
    return(a,b,c)

def FOM(ktot, good_k, nmodes_pk, smoothed_fields, fields, k_fields, angles_array, length_scale,  fom_type, optimiser=True):
    ngrid=Nmesh=256
    
    
    delta_Om=delta_s8=0.02
    
    
    delta_R = smoothed_fields
    
    nmodes=4
    delta_R_fid = smoothed_fields['fiducial']
    print('mean smoothed', np.mean(delta_R_fid))
    delta_R_train = jnp.linspace(np.min(delta_R_fid), 2.0, nmodes).reshape(-1,1) 
    kernel = 20*ConstantKernel(constant_value=1., constant_value_bounds=(0, 30.0))*RBF(length_scale=length_scale,) #this could be changed     
    mark_names=[]
    # tidy this up shouldnt be in this function!
    myvars= globals() ### this var stuff is just a hack for naming variables inside a loop
    
    myvars[f'fft_d_fiducial'] = k_fields['fiducial']
    myvars[f'fft_d_Om_p'] =  k_fields['Om_p']
    myvars[f'fft_d_Om_m'] =  k_fields['Om_m']
    myvars[f'fft_d_s8_p'] =  k_fields['s8_p']
    myvars[f'fft_d_s8_m'] =  k_fields['s8_m']
        
       

    for i in range(1,len(angles_array)+1): #4

        a,b,c = angles_array[i-1]
        w = sin(a)*sin(b)*sin(c)
        x = sin(a)*sin(b)*cos(c)
        y =sin(a)*cos(b)
        z = cos(a)
        mark_nodes= jnp.array([w,x,y,z,])
        myvars[f'mark_nodes{i}'] = jnp.array([w,x,y,z,])
        # vars[f'gpr{i}''] 
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=1)
        gpr.fit(delta_R_train, mark_nodes)
        mark_names.append(f'm{i}')


    #calculate the marks as predicided from gpr
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
        # print(f'{obs}_{f1}{f2}_finite_diff = ', f'Pk_{f1}{f2}_{obs}_p -Pk_{f1}{f2}_{obs}_m') n
#         np.save(f'Pk_{f1}{f2}_{obs}_p', myvars[f'Pk_{f1}{f2}_{obs}_p'])
#         np.save(f'Pk_{f1}{f2}_{obs}_m', myvars[f'Pk_{f1}{f2}_{obs}_m'])
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
            
#             np.save( f'Pk_{n1_a}{n1_b}_fiducial', pk_n1a_n1b, )
#             np.save( f'Pk_{n2_a}{n2_b}_fiducial', pk_n2a_n2b, )
#             np.save( f'Pk_{n1_a}{n2_b}_fiducial', pk_n1a_n2b, )
#             np.save( f'Pk_{n2_a}{n1_b}_fiducial', pk_n2a_n1b, )

            cov[np.ix_(id_a, id_b)] = np.diag((pk_n1a_n1b*pk_n2a_n2b +
                                               pk_n1a_n2b*pk_n2a_n1b)/nmodes_pk)

    pk_d_d = myvars[f'Pk_dd_fiducial']

    cov_dd_dd = np.diag(    2*pk_d_d**2/nmodes_pk) #not sure about this
    deriv_s8_dd = (myvars[f'Pk_dd_s8_p'] - myvars[f'Pk_dd_s8_m'])/(2*delta_s8)
    deriv_Om_dd = (myvars[f'Pk_dd_Om_p'] - myvars[f'Pk_dd_Om_m'])/(2*delta_Om)

#     np.save('covariance_mat_7',cov)
    icov = my_pinv(cov, w_thr=1E-7) #inverse covariance
#     np.save('icov_7', icov)
    fisher_dd = np.array([[jnp.dot(deriv_s8_dd.T, jnp.dot(np.linalg.inv(cov_dd_dd),  deriv_s8_dd)),
                       jnp.dot(deriv_s8_dd.T, jnp.dot(np.linalg.inv(cov_dd_dd), deriv_s8_dd))],
                      [jnp.dot(deriv_Om_dd.T, jnp.dot(np.linalg.inv(cov_dd_dd), deriv_s8_dd)),
                       jnp.dot(deriv_Om_dd.T, jnp.dot(np.linalg.inv(cov_dd_dd), deriv_Om_dd))]])
#     np.save('deriv_s8_7', deriv_s8)
#     np.save('deriv_Om_7', deriv_Om)
    fisher_cov_so = jnp.dot(deriv_s8.T,np.dot( icov, deriv_Om))
    fisher_cov_os = jnp.dot(deriv_Om.T,np.dot( icov, deriv_s8))
    fisher_cov_ss = jnp.dot(deriv_s8.T,np.dot( icov, deriv_s8))
    fisher_cov_oo = jnp.dot(deriv_Om.T,np.dot( icov, deriv_Om))

    fisher_cov = np.array([[fisher_cov_ss, fisher_cov_so],
                           [fisher_cov_os, fisher_cov_oo]])
#     np.save('fisher_cov_7', fisher_cov)
    

    #sanity checks 

    if np.isnan(fisher_cov_os):
        print(f'NAN VALUE for b={b}, p={p}, os')

    if np.isnan(fisher_cov_so):
        print(f'NAN VALUE for b={b}, p={p}, so')

    error = np.sqrt(my_pinv(fisher_cov))
    fid_err =np.sqrt(my_pinv(fisher_dd)) 
#     print('fisher cov',fisher_cov)
    print('error', error)
    print('error improv', fid_err/error)
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

    print('error', error)
    print('fom', fom)
    if optimiser==True:
        #optimiser wants function to have one output only
        return(-fom)
    else:
        return(fom, mark_fid, fisher_cov )

