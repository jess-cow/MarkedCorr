import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.90'
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
import time 

from Pk_tools import Fourier, smooth_field, mark_10, get_Pk
from Pk_tools import np_Fourier, smooth_field, mark_10, np_get_Pk

Fourier_jit =jax.jit(Fourier)

smooth_jit= smooth_field
# jax.jit(smooth_field)
mark_jit = jax.jit(mark_10)
get_Pk_jit = jax.jit(get_Pk)
import jax
jax.default_backend()
jax.devices()

################################################################
# MAIN CODE
################################################################
# setup_logging()
direc = "/mnt/extraspace/damonge/MarkedPk/"

# /mnt/extraspace/damonge/MarkedPk/fiducial_512/Snap/snap_fiducial_512_nside512_seed101_002

names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p',]


#load in using nbodykit
 # fid = Gadget1Catalog(direc+"snap_fiducial_512_nside512_seed101_002")
    # Om_m = Gadget1Catalog(direc+"snap_Om_m_512_nside512_seed101_002")
    # Om_p = Gadget1Catalog(direc+"snap_Om_p_512_nside512_seed101_002")
# s8_m = Gadget1Catalog(direc+"snap_s8_m_512_nside512_seed101_002")
# s8_p = Gadget1Catalog(direc+"snap_s8_p_512_nside512_seed101_002")
    # Om_p = Gadget1Catalog(direc+"snap_Om_p_512_nside512_seed101_002")


delta_Om = 0.02
delta_s8 = 0.02

 
lbox = 700 #Mpc/h
Nmesh = 512 #512
nbins=244
R = 10
p=2
b=0.25

# names = ['s8_m','fiducial', 'Om_m','Om_p', 's8_m','s8_p']



# name = names[0] .
print('s8')
s8m= np.load(f's8_m_arr.npy')
s8p= np.load(f's8_p_arr.npy')

#s8 m
ktot, s8m_k_field = Fourier_jit(s8m)
smoothed_field_s8m = smooth_jit(ktot, s8m_k_field, R)
k, Pk_s8m = get_Pk_jit(s8m_k_field, ktot)
#loop this?
mark_s8m = mark_jit(p, b, smoothed_field_s8m)
marked_field_s8m = s8m_k_field * mark_s8m

k_marked, fft_marked_s8m = Fourier_jit(marked_field_s8m)
k, Pk_md_s8m = get_Pk_jit(fft_marked_s8m, ktot, second = s8m_k_field)


#s8 p
print('s8p')

ktot, s8p_k_field = Fourier_jit(s8p)
smoothed_field_s8p = smooth_jit(ktot, s8p_k_field, R)
k, Pk_s8p = get_Pk_jit(s8p_k_field, ktot)
#loop this?
mark_s8p = mark_jit(p, b, smoothed_field_s8p)
marked_field_s8p = s8p_k_field * mark_s8p

k_marked, fft_marked_s8p = Fourier_jit(marked_field_s8p)
k, Pk_md_s8p = get_Pk_jit(fft_marked_s8p, ktot, second = s8p_k_field)

# name = names[2]
# print(name)
# field_2 = np.load(f'{name}_arr.npy')
# ktot, k_field_2 = Fourier_jit(field_2)
# smoothed_field_2 = smooth_jit(ktot, k_field_2, R)
# k, Pk_2 = get_Pk_jit(k_field_2, ktot)
# mark = mark_jit(p, b, smoothed_field_2)
# marked_field_2 = field_2 * mark
# k_marked, fft_marked_2 = Fourier_jit(marked_field_2)
# k, Pk_md_2 = get_Pk_jit(fft_marked_2, ktot, second = k_field_2)

# name = names[3]
# print(name)
# field_3 = np.load(f'{name}_arr.npy')
# ktot, k_field_3 = Fourier_jit(field_3)
# smoothed_field_3 = smooth_jit(ktot, k_field_3, R)
# k, Pk_3 = get_Pk_jit(k_field_3, ktot)
# marked_field_3 = field_3 * mark
# k_marked, fft_marked_3 = Fourier_jit(marked_field3)
# k, Pk_md_3 = get_Pk_jit(fft_marked_3, ktot, second = k_field_3)




# name = names[4]
# print(name)
# field_4 = np.load(f'{name}_arr.npy')
# ktot, k_field_4 = Fourier_jit(field_4)
# smoothed_field_4 = smooth_jit(ktot, k_field_4, R)
# k, Pk_4 = get_Pk_jit(k_field_4, ktot)
# marked_field_4 = field_4 * mark
# k_marked, fft_marked_2 = Fourier_jit(marked_field_2)
# k, Pk_md_2 = get_Pk_jit(fft_marked_2, ktot, second = k_field_2)

# name = names[5]
# print(name)
# field_5 = np.load(f'{name}_arr.npy')
# ktot, k_field_5 = Fourier_jit(field_5)
# smoothed_field_5 = smooth_jit(ktot, k_field_5, R)
# k, Pk_5 = get_Pk_jit(k_field_5, ktot)

    
# start=time.time()
########
#marking


# names = ['s8_m','fiducial', 'Om_m','Om_p', 's8_m','s8_p']

# # name = names[0]
# # smoothed_field_0 = smoothed_fields[f'{name}']
# # field_0 = fields[f'{name}']
# # k_field_0 = k_fields[f'{name}']
# mark = mark_jit(p, b, smoothed_field_0)
# marked_field = field_0 * mark
# k_marked, fft_marked = Fourier_jit(marked_field)
# k, Pk_mm = get_Pk_jit(fft_marked, ktot)
# k, Pk_md = get_Pk_jit(fft_marked, ktot, second = k_field_0)

# name = names[1]
# smoothed_field_1 = smoothed_fields[f'{name}']
# field_1 = fields[f'{name}']
# k_field_1 = k_fields[f'{name}']
# mark = mark_jit(p, b, smoothed_field_1)
# marked_field = field_1 * mark
# k_marked, fft_marked = Fourier_jit(marked_field)
# k, Pk_mm = get_Pk_jit(fft_marked, ktot)
# k, Pk_md = get_Pk_jit(fft_marked, ktot, second = k_field_1)

# ...

# name = names[5]
# smoothed_field_5 = smoothed_fields[f'{name}']
# field_5 = fields[f'{name}']
# k_field_5 = k_fields[f'{name}']
# mark = mark_jit(p, b, smoothed_field_5)
# marked_field = field_5 * mark
# k_marked, fft_marked = Fourier_jit(marked_field)
# k, Pk_mm = get_Pk_jit(fft_marked, ktot)
# k, Pk_md = get_Pk_jit(fft_marked, ktot, second = k_field_5)

