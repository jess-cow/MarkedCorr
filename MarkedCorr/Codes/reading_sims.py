
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
import jax

from Pk_tools import Fourier, smooth_field, mark_10, get_Pk


Fourier_jit =jax.jit(Fourier)
smooth_jit= jax.jit(smooth_field)
mark_jit = jax.jit(mark_10)
get_Pk_jit = jax.jit(get_Pk)

jax.default_backend()
jax.devices()

################################################################
# MAIN CODE
################################################################
# load in files
direc = "/mnt/extraspace/damonge/MarkedPk/"
names = ['fiducial', 'Om_m', 'Om_p', 's8_m', 's8_p',]

names = ['s8_m','fiducial', 'Om_m','Om_p', 's8_m','s8_p']
seed = '101'
snapnum = '002'
delta_Om = 0.02
delta_s8 = 0.02
sims={}; mesh = {}; power={}; painted_arr={}; fields={}
k_fields={}; smoothed_fields={}; Pks={}

 
lbox = 700 #Mpc/h
Nmesh = 256 #512
nbins=244
R = 10
p=2
b=0.25


for name in names:
    print(name)
    sims[f'{name}'] = Gadget1Catalog(direc+f'{name}_512/Snap/'f"snap_{name}_512_nside512_seed{seed}_{snapnum}")
    mesh = sims[f'{name}'].to_mesh(Nmesh=256) #make mesh
    painted_arr[f'{name}'] = mesh.paint(mode='real') #paint mesh, i.e. apply particles to the grid
    fields[f'{name}'] = jnp.array(painted_arr[f'{name}'])
    del painted_arr[f'{name}']; del sims[f'{name}'] 
    np.save(f'/mnt/zfsusers/jesscowell/MarkedCorr/Data/Sim_arrays/{name}_{Nmesh}_arr', fields[f'{name}'])
    # ktot, k_fields[f'{name}']= Fourier_jit(fields[f'{name}'])
    #smooth the field 
    # smoothed_fields[f'{name}'] = smooth_jit(ktot, k_fields[f'{name}'], R)
    #calculate the Pk of orignal field 
    # k, Pks[f'{name}'] = get_Pk_jit(k_fields[f'{name}'], ktot)

    