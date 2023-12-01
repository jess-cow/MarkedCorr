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


class Mark(object):
    def __init__(self, angles, kmax=0.3, kmin=0.01, fom_type='total', lbox=700., ngrid=256, 
                 n_nodes=4, l_gp=2.0, A_gp=10.0, jitter_gp=1E-3, w_thr=1E-7, R=10, prefix=''):
    
        self.angles=angles
        self.nodes = self.convert_from_angle(self.angles)
        
        # Eigenvalue threshold, for condition number ratio in pseudo inverse
        self.w_thr = w_thr
        self.smoothing_scale=R
        # k range (h/Mpc)
        self.kmin = kmin
        self.kmax = kmax
        # Quantity to maximize
        self.fom = fom_type #both, om or s8
        # self.delta_range = [-1.0, 5.0] #davids? not sure
        # Parameter intervals for finite differences
        self.dpar = {'Om': 0.02, 's8': 0.02}
        # Simulation names
        self.sims = ["fid", "Om_m", "Om_p", "s8_m", "s8_p"]
        #load in data
        self.ktot, self.good_k, self.Pks, self.fields, self.k_fields,self.smoothed_fields, self.nmodes_pk = self.load_sims(R=self.smoothing_scale)
        #fisher of pure Pk
        
        self.delta_s8 = 0.02
        self.delta_om =0.02
        # Box length (Mpc/h)
        self.lbox = lbox
        # Grid size
        self.ngrid = ngrid
        # GP lengthscale in delta
        self.l_gp = l_gp
        # GP amplitude
        self.A_gp = A_gp
        
        self.Pk_fisher, self.Pk_cov = self.get_Pk_only_fisher()
        
        # # GP jitter
        # self.jitter_gp = jitter_gp
    def smooth_field(self, k, field, R):
        '''
        Smooth a field by a given radius R(Mpc).
        Field supplied should be in k space

        '''
        W =  jnp.exp(-0.5*(k*R)**2)
        smoothed_field = (W* field)
        return smoothed_field
    def Fourier(self, field, Nmesh = 256, lbox=700, inverse=False):
        '''Return FFT of field
        Parameters:
        field - the 3D field array 
        lbox - Length of simulation in Mpc/h
        Nmesh - size of mesh
        kmin - min wave number to calculate Pk at
        nbins - number of k bins to use
        
        Returns:
        k-centers - grid of k values
        k-field - FFT of real field  
        '''
        # cell width
        d = lbox / Nmesh
        # Take the Fourier Transform of the 3D box
        if inverse:
            complex_field = jnp.fft.irfftn(field, )
        else:
            complex_field = jnp.fft.rfftn(field, )
        # natural wavemodes 
        kx = ky = jnp.fft.fftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
        kz = jnp.fft.rfftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
        nx, ny, nz = complex_field.shape #shape of ft is symmetric 
        # Compute the total wave number
        ktot = jnp.sqrt(kx[:,None, None]**2 + ky[None, :, None]**2+kz[None,None,:]**2)[:nx, :ny, :nz]
        if np.isnan(complex_field).any():
            print('fourier transform is nan!')
            quit()
        return ktot, complex_field
    def load_sims(self, R=10):
        
        
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

        fields={}; rho_fields={}
        k_fields={}; smoothed_kfields={}; Pks={}; smoothed_fields={}
        delta_R={}
        for name in names:
            #load in the raw density field
            rho_fields[f'{name}'] = np.load(f'/mnt/extraspace/jesscowell/MarkedCorr/Data/Sim_arrays/{name}_{Nmesh}_arr.npy')
            #calculate the overdensity field
            fields[f'{name}'] =  rho_fields[f'{name}']/np.mean(rho_fields[f'{name}']) -1
            #fourier transform overdensity field
            ktot, k_fields[f'{name}']= self.Fourier(fields[f'{name}'], Nmesh=Nmesh)
            #smooth fields in k space
            smoothed_kfields[f'{name}'] = self.smooth_field(ktot, k_fields[f'{name}'], R, ) 
            #smoothed fields in real space
            _, smoothed_fields[f'{name}'] = self.Fourier(smoothed_kfields[f'{name}'], inverse=True)
            #smoothed overdensity in real space, 
            delta_R[f'{name}']=smoothed_fields[f'{name}']
            #get power spectrum of og field
            k, nmodes_pk, pk = get_Pk(k_fields[f'{name}'], ktot)
            #scale cuts
            good_k =  k < kmax 
            k = k[good_k][3:] #throw away first 3 points also
            Pks[f'{name}'] = pk[good_k][3:]
            nmodes_pk = nmodes_pk[good_k][3:]
        

        return(ktot,good_k, Pks, fields, k_fields,smoothed_fields, nmodes_pk)
    def get_Pk_only_fisher(self,):
        Pks=self.Pks
        deriv_s8 = (Pks['s8_p'] - Pks['s8_m'])/(2*self.delta_s8)
        deriv_Om = (Pks['Om_p'] - Pks['Om_m'])/(2*self.delta_s8)
        cov = np.diag(2*(Pks['fiducial']**2)/self.nmodes_pk)
        icov = self.my_pinv(cov)
        fisher_cov_so = jnp.dot(deriv_s8.T,np.dot( icov, deriv_Om))
        fisher_cov_os = jnp.dot(deriv_Om.T,np.dot( icov, deriv_s8))
        fisher_cov_ss = jnp.dot(deriv_s8.T,np.dot( icov, deriv_s8))
        fisher_cov_oo = jnp.dot(deriv_Om.T,np.dot( icov, deriv_Om))
        
        fisher_cov = np.array([[fisher_cov_ss, fisher_cov_so],
                            [fisher_cov_os, fisher_cov_oo]])
        print('deriv_8', deriv_s8)
        print('deriv_Om', deriv_Om)
        print('fid Pk', Pks['fiducial'])
        
        return(fisher_cov, cov)
    
    def pairs(self, mark_names, names):
        '''Generates all possible pairings for calculating Pks. Give mark names e.g. (m1, m2, m3), and names for all fields used, e.g. names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p']
    NB when doing finite diff need Om_m etc, bit for general covariance just need fiducial.
     {mark},{mark},{simulation} '''
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


    def iterate_pairs(self, mark_names):
        '''
        index and pairing d x m1 etc, 
        '''
        map_names = ['d'] + mark_names
        ipk = 0
        for i1, n1 in enumerate(map_names):
            for n2 in map_names[i1:]: 
                yield ipk, n1, n2
                ipk += 1
        


        
    def my_pinv(self, cov, w_thr=1E-7):
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
    def convert_to_angle(self, points):
        #check this??
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
            

        # w = sin(a)*sin(b)*sin(c)
        # x = sin(a)*sin(b)*cos(c)
        # y =sin(a)*cos(b)
        # z = cos(a)
        
        return(a,b,c)

    def convert_from_angle(self, angles):
        '''convert from 3 angles to 4 points in delta_R space  i.e. cartesian 4D on surface of sphere'''
        a,b,c = angles
                    
        w = sin(a)*sin(b)*sin(c)
        x = sin(a)*sin(b)*cos(c)
        y =sin(a)*cos(b)
        z = cos(a)
        return(np.array([w,x,y,z]))
        
    def get_mark_from_nodes(self, angles_array):
        #takes array of marks
        
        ngrid=Nmesh=256
        kernel = 20*ConstantKernel(constant_value=1., constant_value_bounds=(0, 30.0))*RBF(length_scale=length_scale,) #this could be changed     
        nmodes=4
        delta_R_fid = self.smoothed_fields['fiducial']
        print('mean smoothed', np.mean(delta_R_fid))
        delta_R_train = jnp.linspace(np.min(delta_R_fid), 2.0, nmodes).reshape(-1,1) 
        mark_names=[]
        # tidy this up shouldnt be in this function!
        myvars= globals() ### this var stuff is just a hack for naming variables inside a loop
        
        myvars[f'fft_d_fiducial'] = k_fields['fiducial']
        myvars[f'fft_d_Om_p'] =  k_fields['Om_p']
        myvars[f'fft_d_Om_m'] =  k_fields['Om_m']
        myvars[f'fft_d_s8_p'] =  k_fields['s8_p']
        myvars[f'fft_d_s8_m'] =  k_fields['s8_m']
            
       
        marks_dict={}
        for i in range(1,len(angles_array)+1): #4
            angles = angles_array[i-1]
            w, x, y, z = self.covert_from_angle(self, angles)

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
            marks_dict[f'mark_fid_{i}'] = mark_fid.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_Omp{i}'] = mark_Omp.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_Omm{i}']= mark_Omm.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_s8_p{i}'] = mark_s8p.reshape([ngrid, ngrid, ngrid])
            marks_dict[f'mark_s8_m{i}']= mark_s8m.reshape([ngrid, ngrid, ngrid])

            ################################
            #calculate the marked field in real space 

            marked_field_fid = self.fields[f'fiducial']*marks_dict[f'mark_fid_{i}']
            marked_field_omp = self.fields[f'Om_p']*marks_dict[f'mark_Omp{i}']
            marked_field_omm = self.fields[f'Om_m']*marks_dict[f'mark_Omm{i}']
            marked_field_s8p =self.fields[f's8_p']*marks_dict[f'mark_s8_p{i}']
            marked_field_s8m = self.fields[f's8_m']*marks_dict[f'mark_s8_m{i}']

            #fft marked field
            _, myvars[f'fft_m{i}_fiducial'] = Fourier(marked_field_fid, Nmesh=Nmesh)
            _, myvars[f'fft_m{i}_Om_p'] = Fourier(marked_field_omp, Nmesh=Nmesh)
            _, myvars[f'fft_m{i}_Om_m'] = Fourier(marked_field_omm, Nmesh=Nmesh)
            _, myvars[f'fft_m{i}_s8_p'] = Fourier(marked_field_s8p, Nmesh=Nmesh)
            _, myvars[f'fft_m{i}_s8_m'] = Fourier(marked_field_s8m, Nmesh=Nmesh)
            
        
            
        return(marks_dict)
    

    def get_all_Pks(self,):
        Pks={}
        for ix, f1, f2, f3 in self.pairs(): 
             #calculate all Pk for diff fields m1dm, d2d etc, 
             _, _, myvars[f'Pk_{f1}{f2}_{f3}'] =  get_Pk(myvars[f'fft_{f1}_{f3}'], ktot, second =myvars[f'fft_{f2}_{f3}'])
             #scale cuts
             Pks[f'Pk_{f1}{f2}_{f3}'] =Pks[f'Pk_{f1}{f2}_{f3}'][good_k][3:]
             return(Pks)





    def get_fom(self, mark_names, fom_type ):
        '''get fom from mark function'''
        #calculate Pks
        Pks = self.get_all_Pks()
        #calculate derivatives
        derivs = self.get_derivs(Pks)
        #calculate theoretical covariance
        cov = self.get_cov(Pks, mark_names)
        #inverse cov
        icov = self.my_pinv(cov, w_thr=1E-7) #inverse covariance
        #fisher matrix 
        deriv_s8 = derivs['deriv_s8']; deriv_Om = derivs['deriv_Om']
        

        fisher_cov_so = jnp.dot(deriv_s8.T,np.dot( icov, deriv_Om))
        fisher_cov_os = jnp.dot(deriv_Om.T,np.dot( icov, deriv_s8))
        fisher_cov_ss = jnp.dot(deriv_s8.T,np.dot( icov, deriv_s8))
        fisher_cov_oo = jnp.dot(deriv_Om.T,np.dot( icov, deriv_Om))

        fisher_cov = np.array([[fisher_cov_ss, fisher_cov_so],
                            [fisher_cov_os, fisher_cov_oo]])
        
        if fom_type =='both':
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


    def get_derivs(self, Pks):
        derivs={}
        for ix, f1, f2, obs in finite_diff_pair():
            derivs[f'{obs}_{f1}{f2}_finite_diff'] = (Pks[f'Pk_{f1}{f2}_{obs}_p'] - Pks[f'Pk_{f1}{f2}_{obs}_m'])/(2*self.delta_Om)
            derivs[f'deriv_{obs}'] = (np.hstack([ derivs[f'deriv_{obs}'],derivs[f'{obs}_{f1}{f2}_finite_diff'] ]))
        return derivs

    def get_cov(self, Pks, mark_names, marked=True):
        length = comb(len(mark_names)+2,2)
        ndata = len(deriv_s8)
        idx=[]
        
        i=0 
        indices={}
        for ix, n1, n2, in self.iterate_pairs():
            idx.append(f'id_{n1}{n2}' )
            indices[f'{n1}{n2}'] =  np.arange(ndata).reshape([length, ndata//length])[i]
            i+=1

        cov = np.zeros([ndata, ndata])
    
        
        for ipka, n1_a, n2_a in self.iterate_pairs():
            id_a = indices[f'{n1_a}{n2_a}']
            for ipkb, n1_b, n2_b in self.iterate_pairs():
                id_b = indices[f'{n1_b}{n2_b}']
                pk_n1a_n1b = Pks[f'Pk_{n1_a}{n1_b}_fiducial']
                pk_n2a_n2b = Pks[f'Pk_{n2_a}{n2_b}_fiducial']
                pk_n1a_n2b = Pks[f'Pk_{n1_a}{n2_b}_fiducial']
                pk_n2a_n1b = Pks[f'Pk_{n2_a}{n1_b}_fiducial']
            
#             np.save( f'Pk_{n1_a}{n1_b}_fiducial', pk_n1a_n1b, )
#             np.save( f'Pk_{n2_a}{n2_b}_fiducial', pk_n2a_n2b, )
#             np.save( f'Pk_{n1_a}{n2_b}_fiducial', pk_n1a_n2b, )
#             np.save( f'Pk_{n2_a}{n1_b}_fiducial', pk_n2a_n1b, )

                cov[np.ix_(id_a, id_b)] = np.diag((pk_n1a_n1b*pk_n2a_n2b +
                                                pk_n1a_n2b*pk_n2a_n1b)/self.nmodes_pk)
        return cov
