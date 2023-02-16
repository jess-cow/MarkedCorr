import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import nbodykit
import numpy as np
from nbodykit.lab import *
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import time
from Pk_tools import Fourier, smooth_field, mark_10, get_Pk


#information for loading in sims
names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p',]
seed = '101'; snapnum = '002'


R = 10 #smoothing in Mpc
Nmesh=256 #resolution
kmax = 1.0 #max k for scale cuts 
delta_Om =  delta_s8 = 0.02

sims={}; mesh = {}; power={}; painted_arr={}; fields={}
k_fields={}; smoothed_kfields={}; Pks={}; smoothed_fields={}

for name in names:
    print(name)
    # TODO: Make sure this has the right resolution
    fields[f'{name}'] = np.load(f'{name}_arr.npy')
    print(np.shape((fields[f'{name}'])))

    ktot, k_fields[f'{name}']= Fourier(fields[f'{name}'], Nmesh=Nmesh)
    print(np.shape(k_fields[f'{name}']))
    print('kfields',np.shape(k_fields[f'{name}']), 'ktot', np.shape(ktot))
    
    smoothed_kfields[f'{name}'] = smooth_field(ktot, k_fields[f'{name}'], R, ) #IN K SPACE
    print(np.shape(smoothed_kfields[f'{name}']), 'smoooooth')

    # TODO: Check normalisation of ifft
    _, smoothed_fields[f'{name}'] = Fourier(smoothed_kfields[f'{name}'], inverse=True)
    print(np.shape(smoothed_fields[f'{name}']))
    print(smoothed_fields[f'{name}'][2, 16, 12])

    k, nmodes_pk, pk = get_Pk(k_fields[f'{name}'], ktot)
    good_k = k < kmax
    print(len(good_k), np.sum(good_k))
    k = k[good_k]
    nmodes_pk = nmodes_pk[good_k]
    Pks[f'{name}'] = pk[good_k]

om_finite_diff =(Pks['Om_p']- Pks['Om_m'])/(2*delta_Om)
s8_finite_diff =(Pks['s8_p']- Pks['s8_m'])/(2*delta_s8)

    

#############################################
# CALCULATING FOR DIFFERENT MARKS
#############################################

fishers = []
p_arr=[]
b_arr=[]
fisher_direc='/mnt/zfsusers/jesscowell/MarkedCorr/Data/Fisher_info/'

for b in jnp.linspace(-0.1, 2.0, 10): # delta_s
    start=time.time()
    for p in jnp.linspace(0.1, 9 , 10): #p in mark
        print(b, p)
        # Compute marks
        mark_fid = mark_10(p, b, smoothed_fields[f'fiducial'])
        mark_Omp = mark_10(p, b, smoothed_fields[f'Om_p'])
        mark_Omm = mark_10(p, b, smoothed_fields[f'Om_m'])
        mark_s8m = mark_10(p, b, smoothed_fields[f's8_m'])
        mark_s8p = mark_10(p, b, smoothed_fields[f's8_p'])
        #calculate the marked field
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

        #scale cuts 
        Pk_dd_fid = Pks['fiducial']
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

        # calculate fisher 
        fisher_dd = np.array([[np.dot(s8_finite_diff, np.dot(np.linalg.inv(cov_dd_dd), s8_finite_diff)),
                               np.dot(s8_finite_diff, np.dot(np.linalg.inv(cov_dd_dd), om_finite_diff))],
                              [np.dot(om_finite_diff, np.dot(np.linalg.inv(cov_dd_dd), s8_finite_diff)),
                               np.dot(om_finite_diff, np.dot(np.linalg.inv(cov_dd_dd), om_finite_diff))]])
        fisher_cov_so = np.dot(deriv_s.T,np.dot( np.linalg.inv(cov), deriv_o))
        fisher_cov_os = np.dot(deriv_o.T,np.dot( np.linalg.inv(cov), deriv_s))
        fisher_cov_ss = np.dot(deriv_s.T,np.dot( np.linalg.inv(cov), deriv_s))
        fisher_cov_oo = np.dot(deriv_o.T,np.dot( np.linalg.inv(cov), deriv_o))

        fisher_cov = 0.5*np.array([[fisher_cov_ss, fisher_cov_so],[fisher_cov_os, fisher_cov_oo]])

        cov_mark = np.linalg.inv(fisher_cov)
        r_om_s8_mark = cov_mark[0, 1]/np.sqrt(cov_mark[0, 0]*cov_mark[1, 1])
        cov_dd = np.linalg.inv(fisher_dd)
        r_om_s8_dd = cov_dd[0, 1]/np.sqrt(cov_dd[0, 0]*cov_dd[1, 1])
        print(np.sqrt(np.diag(cov_mark)), r_om_s8_mark)
        print(np.sqrt(np.diag(cov_dd)), r_om_s8_dd)
        np.save(fisher_direc+f'b={b}_p={p}', fisher_cov)
        fishers.append(fisher_cov)
        b_arr.append(b)
        p_arr.append(p)
        
 

