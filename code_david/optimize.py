import numpy as np
from scipy.optimize import minimize
import sys
import marker

seed = int(sys.argv[1])
kmax = float(sys.argv[2])
fom_type = sys.argv[3]
outdir = sys.argv[4]
n_nodes = int(sys.argv[5])

print(seed)
np.random.seed(seed)
mk = marker.Marker(kmax=kmax, fom_type=fom_type, n_nodes=n_nodes)
ang0 = np.array(mk.get_random_angles())
r = minimize(lambda x: -mk.get_fom(x), ang0, method='Powell', options={'ftol': 0.01})
np.savetxt(f"{outdir}/seed{seed}_angles.txt", r.x)
