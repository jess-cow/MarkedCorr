import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import nbodykit
import numpy as np
from nbodykit.lab import *
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib
import os
# import time

def Fourier(field, Nmesh = 512, lbox=700,):
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
    complex_field = jnp.fft.fftn(field)
     # natural wavemodes 
    kx =ky = kz = jnp.fft.fftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
     # Compute the total wave number
    ktot = jnp.sqrt(kx[:,None, None]**2 + ky[None, :, None]**2+kz[None,None,:]**2)

    return ktot, complex_field


def np_Fourier(field, Nmesh = 512, lbox=700,):
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
    complex_field = np.fft.fftn(field)
     # natural wavemodes 
    kx =ky = kz = np.fft.fftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
     # Compute the total wave number
    ktot = np.sqrt(kx[:,None, None]**2 + ky[None, :, None]**2+kz[None,None,:]**2)

    return ktot, complex_field


def mark_10(p, b, delta_R):
    '''calculates the mark with smoothing scale of 10MPc
    Parameters:
    p
    b
    delta_R : smoothed field
    '''
    m = (1 + delta_R/(1+b))**(-p)
    return(m)

# function optimized to run on gpu 

           
def get_Pk(fourier_data, ktot, second=None,lbox = 700, Nmesh = 512, nbins=244, kmin=0.01):
    """
    Get 1D power spectrum from an already painted field. <- need to do this without NBodykit?
    
    
    Parameters:
     field - the 3D field array 
     lbox - Length of simulation in Mpc/h
     Nmesh - size of mesh
     kmin - min wave number to calculate Pk at
     nbins - number of k bins to use
     
    Returns:
     k-centers - 1D array of central k-bin values
     Pk - 1D array of k at each k 
      
    """
    

    if second is None:
        second = fourier_data
    else:
        print('calculating cross Pk')
   
    # Compute the squared magnitude of the complex numbers in the Fourier space, will give 3D Pk array
    power_spectrum =  (fourier_data)*jnp.conjugate(second) #abs changes value here!


    # cell width
    d = lbox / Nmesh

    # nyquist frequency k=pi*Nmesh/Lbox, will be the max k in k binning 
    kN = jnp.pi / d


    # # natural wavemodes 
    # kx =ky = kz = jnp.fft.fftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
    # # Compute the total wave number
    # ktot = jnp.sqrt(kx[:,None, None]**2 + ky[None, :, None]**2+kz[None,None,:]**2)

    #bin the k to find the total at each |k|
    n_cells, k_bins= jnp.histogram(ktot, bins=nbins,range=[kmin, kN] )
    
    #power spectrum is average so weight by each 3D Pk value and sum,
    sum_pk, k_bins= jnp.histogram(ktot, bins=244,range=[kmin, kN], weights=power_spectrum)  #range=[0.01, 10]

    
    pk = sum_pk/n_cells #then divide by number averaged over

    #find center of k bins
    k_center =  (k_bins[1:] + k_bins[:-1])*.5

    vol =(lbox)**3 /(2*jnp.pi)#in (Mpc/h)^3 #not sure why need 2pi but we do 
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)  
    return k_center, pk/vol

def np_get_Pk(fourier_data, ktot, second=None,lbox = 700, Nmesh = 512, nbins=244, kmin=0.01):
    """
    Get 1D power spectrum from an already painted field. <- need to do this without NBodykit?
    
    
    Parameters:
     field - the 3D field array 
     lbox - Length of simulation in Mpc/h
     Nmesh - size of mesh
     kmin - min wave number to calculate Pk at
     nbins - number of k bins to use
     
    Returns:
     k-centers - 1D array of central k-bin values
     Pk - 1D array of k at each k 
      
    """
    

    if second is None:
        second = fourier_data
    else:
        print('calculating cross Pk')
   
    # Compute the squared magnitude of the complex numbers in the Fourier space, will give 3D Pk array
    power_spectrum =  (fourier_data)*jnp.conjugate(second) #abs changes value here!


    # cell width
    d = lbox / Nmesh

    # nyquist frequency k=pi*Nmesh/Lbox, will be the max k in k binning 
    kN = np.pi / d


    # # natural wavemodes 
    # kx =ky = kz = jnp.fft.fftfreq(Nmesh, d=d)* 2. * np.pi # h/Mpc
    # # Compute the total wave number
    # ktot = jnp.sqrt(kx[:,None, None]**2 + ky[None, :, None]**2+kz[None,None,:]**2)

    #bin the k to find the total at each |k|
    n_cells, k_bins= np.histogram(ktot, bins=nbins,range=[kmin, kN] )
    
    #power spectrum is average so weight by each 3D Pk value and sum,
    sum_pk, k_bins= np.histogram(ktot, bins=244,range=[kmin, kN], weights=power_spectrum)  #range=[0.01, 10]

    
    pk = sum_pk/n_cells #then divide by number averaged over

    #find center of k bins
    k_center =  (k_bins[1:] + k_bins[:-1])*.5

    vol =(lbox)**3 /(2*np.pi)#in (Mpc/h)^3 #not sure why need 2pi but we do 
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)  
    return k_center, pk/vol

def smooth_field(k, field, R):
    '''
    Smooth a field by a given radius R(Mpc).
    Field supplied should be in k space
    
    '''
    W =  jnp.exp(-0.5*k*R**2)
    smoothed_field = W* field
    # jnp.inner(W, field)
    return smoothed_field

