#!/bin/bash

kmax=0.1
fom_type='total'
n_nodes=4
outdir='k0p1_optTotal'

mkdir -p ${outdir}
for seed in {1001..1100}
do
    echo ${seed}
    addqueue -c 1day -n 1x12 -s -q cmb -m 1 /users/damonge/miniconda3/envs/condaenv/bin/python3 optimize.py ${seed} ${kmax} ${fom_type} ${outdir} ${n_nodes}
done
