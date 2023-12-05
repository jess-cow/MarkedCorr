import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize


class Marker(object):
    def __init__(self, kmax=0.3, kmin=0.01, fom_type='total', lbox=700., ngrid=256,
                 n_nodes=4, l_gp=2.0, A_gp=10.0, jitter_gp=1E-3, w_thr=1E-7, prefix='',
                 fix_origin=False):
        self.prefix = prefix
        # Eigenvalue threshold
        self.w_thr = w_thr
        # k range (h/Mpc)
        self.kmin = kmin
        self.kmax = kmax
        # Quantity to maximize
        self.fom = fom_type
        self.delta_range = [-1.0, 5.0]
        # Parameter intervals for finite differences
        self.dpar = {'Om': 0.02, 's8': 0.02}
        # Simulation names
        self.sims = ["fid", "Om_m", "Om_p", "s8_m", "s8_p"]
        # Box length (Mpc/h)
        self.lbox = lbox
        # Grid size
        self.ngrid = ngrid
        # GP lengthscale in delta
        self.l_gp = l_gp
        # GP amplitude
        self.A_gp = A_gp
        # GP jitter
        self.jitter_gp = jitter_gp
        self.fix_origin = fix_origin

        # Overdensity field nodes, GP covariance and associated GP filter
        self.n_nodes = n_nodes
        self.delta_nodes = (np.arange(self.n_nodes)+0.5)/self.n_nodes*(self.delta_range[1]-self.delta_range[0])+self.delta_range[0]
        if self.fix_origin:
            if 0.0 not in self.delta_nodes:
                self.delta_nodes = np.sort(np.concatenate((self.delta_nodes, np.array([0.0]))))
            self.not0 = self.delta_nodes != 0.0
        self.delta_hires = np.linspace(self.delta_range[0], self.delta_range[1], 128)
        self.inv_cov_nodes = np.linalg.inv(self.cov_gp(self.delta_nodes, self.delta_nodes))
        self.cov_nodes_hires = self.cov_gp(self.delta_hires, self.delta_nodes)
        self.filter_gp = np.dot(self.cov_nodes_hires, self.inv_cov_nodes)
    
        # k sampling
        self.kfull = np.fft.fftfreq(self.ngrid, d=self.lbox/self.ngrid)*2*np.pi
        self.khalf = np.fft.rfftfreq(self.ngrid, d=self.lbox/self.ngrid)*2*np.pi
        self.ks = np.sqrt(self.kfull[:, None, None]**2+
                          self.kfull[None, :, None]**2+
                          self.khalf[None, None, :]**2)
        self.ncell, self.kb = np.histogram(self.ks.flatten(), bins=self.ngrid//2,
                                           range=[0, self.ngrid*np.pi/self.lbox])
        self.km = 0.5*(self.kb[1:]+self.kb[:-1])
        self.good_k = (self.km < self.kmax) & (self.km > self.kmin)
        self.num_k = np.sum(self.good_k)
    
        # Read density grids
        self.dens = {n: self.read_grid(self.prefix+n+"_dens") for n in self.sims}
        self.dens_sm = {n: self.read_grid(self.prefix+n+"_dens_sm") for n in self.sims}
    
        # Transform density field to fourier space and compute unmarked Pks
        self.dens_k = {n: self.get_ft(self.dens[n]) for n in self.sims}

    def cov_gp(self, x, y):
        # GP covariance
        jit = self.jitter_gp**2 * (x[:, None] == y[None, :])
        sm = self.A_gp*np.exp(-0.5*((x[:, None] - y[None, :])/self.l_gp)**2)
        return sm + jit

    def read_grid(self, prefix, nfiles=1, verbose=False):
        # Reads density grid from DensTools format
        # Returns grid as 3D array
        f=open(prefix+".0000","rb")
        num_grids, ngrid=np.fromfile(f,dtype=np.int32,count=2)
        assert ngrid == self.ngrid
        f.close()
    
        if verbose:
            print("Will read %d fields"%num_grids+" with %d^3 nodes"%ngrid)
        grid_out=np.zeros([ngrid,ngrid,ngrid,num_grids])
        for ifil in np.arange(nfiles) :
            f=open(prefix+".%04d"%ifil,"rb")
            nug,ng=np.fromfile(f,dtype=np.int32,count=2)
            if (nug!=num_grids) or (ng!=ngrid) :
                print("shit")
                sys.exit(1)
            nx_here=np.fromfile(f,dtype=np.int32,count=1)[0]
            if verbose:
                print("File #%d"%(ifil+1)+", %d slices found"%nx_here)
            for ix in np.arange(nx_here) :
                ix_this=np.fromfile(f,dtype=np.int32,count=1)[0]
                grid_out[ix_this,:,:,:]=np.fromfile(f,dtype=np.float32,count=ng*ng*nug).reshape([ng,ng,nug])
            f.close()
    
        if num_grids==1 :
            grid_out=grid_out[:,:,:,0]
    
        return grid_out

    def get_ft(self, grid):
        # FFT only if needed
        if grid.dtype == 'float64':
            return np.fft.rfftn(grid)*(self.lbox/self.ngrid)**3
        else:
            return grid

    def get_pk(self, grid1, grid2):
        # P(k) between two grids
        dk1 = self.get_ft(grid1)
        dk2 = self.get_ft(grid2)
    
        # P(k) estimator
        sm, kb = np.histogram(self.ks.flatten(), bins=self.ngrid//2,
                              range=[0, self.ngrid*np.pi/self.lbox],
                              weights=np.real(dk1*np.conjugate(dk2)).flatten())
        pk = sm/(self.ncell*self.lbox**3)
        return pk

    def get_mark_from_nodes(self, nodes, kind='gp'):
        # Return mark function (as callable) from a set of nodes
        r = np.sqrt(np.sum(nodes**2))
        if self.fix_origin:
            nod = np.zeros_like(self.delta_nodes)
            nod[self.not0] = nodes
            nodes = nod
        if kind == 'spline':
            f = interp1d(self.delta_nodes, nodes/r, fill_value='extrapolate',
                         kind='cubic', bounds_error=False)
        elif kind == 'gp':
            mark_hires = np.dot(self.filter_gp, nodes/r)
            f = interp1d(self.delta_hires, mark_hires, kind='linear')
        else:
            raise KeyError(f"Unknonwn type {kind}")

        # Force to go through origin
        offset = f(0.0)
        return lambda x : f(x)-offset

    def get_nodes_from_angles(self, angs):
        # Get node values from angles
        nodes = np.ones(self.n_nodes)
        for i in range(self.n_nodes):
            if i > 0:
                nodes[i] = np.cos(angs[i-1])
            for j in range(i, self.n_nodes-1):
                nodes[i] *= np.sin(angs[j])
        return nodes

    def get_angles_from_nodes(self, nodes):
        n = nodes/np.sqrt(np.sum(nodes**2))
        angs = np.zeros(self.n_nodes-1)
        for i in range(self.n_nodes-1):
            if i == 0:
                n0 = n[0]
            else:
                n0 = np.sqrt(np.sum(n[:i+1]**2))
            angs[i] = np.arctan2(n0, n[i+1])
        return angs

    def get_mark_from_angles(self, angs):
        # Return mark function (as callable) from angles
        nodes = self.get_nodes_from_angles(angs)
        return self.get_mark_from_nodes(nodes)

    def get_random_angles(self):
        n = np.random.randn(self.n_nodes)
        n = n/np.sqrt(np.sum(n**2))
        return self.get_angles_from_nodes(n)

    def get_marked_field(self, delta, delta_sm, mark):
        # Apply mark to smoothed field and apply to unsmoothed field
        # Mark
        mk = mark(delta_sm)
        # m = mark * (1+delta) - < mark * (1+delta) >
        den = 1+delta
        marked = den*mk
        marked -= np.mean(marked)
        return marked

    def my_pinv(self, cov):
        if self.w_thr is None:
            return np.linalg.inv(cov)

        w, v = np.linalg.eigh(cov)
        inv_cond = w/np.max(w)
        badw = inv_cond < self.w_thr
        w_inv = 1./w
        w_inv[badw] = 0.
        pinv = np.dot(v, np.dot(np.diag(w_inv), v.T))
        return pinv

    def inspect_mark(self, mark):
        # Get all marked fields and their ffts
        marked = {n: self.get_marked_field(self.dens[n], self.dens_sm[n], mark)
                  for n in self.sims}
        marked_k = {n: self.get_ft(marked[n]) for n in self.sims}

        # Get all pks:
        pks = {}
        for nm1, fld1 in zip(['d', 'm'], [self.dens_k, marked_k]):
            for nm2, fld2 in zip(['d', 'm'], [self.dens_k, marked_k]):
                nm_pk = nm1+nm2
                if nm_pk == 'md':
                    pks['md'] = pks['dm']
                else:
                    pks[nm_pk] = {n: self.get_pk(fld1[n], fld2[n])[self.good_k]
                                  for n in self.sims}

        # Get pk derivatives
        pk_order = ['dd', 'dm', 'mm']
        dpk = {par: {nm_pk: (pks[nm_pk][par+'_p']-pks[nm_pk][par+'_m'])/(2*self.dpar[par])
                     for nm_pk in pk_order}
               for par in ['Om', 's8']}

        # Get covariance
        cov = {}
        for nm_pk1 in pk_order:
            nm1_a, nm1_b = nm_pk1[0], nm_pk1[1]
            for nm_pk2 in pk_order:
                nm2_a, nm2_b = nm_pk2[0], nm_pk2[1]
                pk_1a_2a = pks[nm1_a+nm2_a]['fid']
                pk_1b_2b = pks[nm1_b+nm2_b]['fid']
                pk_1a_2b = pks[nm1_a+nm2_b]['fid']
                pk_1b_2a = pks[nm1_b+nm2_a]['fid']
                covar = (pk_1a_2a*pk_1b_2b+pk_1a_2b*pk_1b_2a)/self.ncell[self.good_k]
                cov[nm_pk1 + '-' + nm_pk2] = np.diag(covar)

        # Form actual covariance and data vector
        d_pk = {par: dpk[par]['dd'] for par in ['Om', 's8']}
        d_all = {par: np.array([dpk[par][nm_pk] for nm_pk in pk_order]).flatten() for par in ['Om', 's8']}
        covar = np.array([[cov[nm_pk1+'-'+nm_pk2] for nm_pk2 in pk_order] for nm_pk1 in pk_order])
        covar = np.transpose(covar, axes=[0, 2, 1, 3])
        cov_pk = covar[0, :, 0, :]
        cov_all = covar.reshape([3*self.num_k, 3*self.num_k])

        # Invert covariance
        inv_cov_all = self.my_pinv(cov_all)
        inv_cov_pk = self.my_pinv(cov_pk)
        
        # Fisher matrix for Pk
        Fpk = np.array([
            [np.dot(d_pk['Om'], np.dot(inv_cov_pk, d_pk['Om'])),
             np.dot(d_pk['Om'], np.dot(inv_cov_pk, d_pk['s8']))],
            [np.dot(d_pk['s8'], np.dot(inv_cov_pk, d_pk['Om'])),
             np.dot(d_pk['s8'], np.dot(inv_cov_pk, d_pk['s8']))]])
        Fall = np.array([
            [np.dot(d_all['Om'], np.dot(inv_cov_all, d_all['Om'])),
             np.dot(d_all['Om'], np.dot(inv_cov_all, d_all['s8']))],
            [np.dot(d_all['s8'], np.dot(inv_cov_all, d_all['Om'])),
             np.dot(d_all['s8'], np.dot(inv_cov_all, d_all['s8']))]])

        return {'pks': np.array([pks[nm_pk]['fid'] for nm_pk in pk_order]),
                'derivs': {n: d_all[n].reshape([3, self.num_k]) for n in ['Om', 's8']},
                'covar': covar, 'fish_pk': Fpk, 'fish_all': Fall}

    def get_fom_from_inspection(self, mdir):
        if self.fom == 'total':
            f = np.linalg.det(mdir['fish_all'])
        elif self.fom == 'Om':
            f = mdir['fish_all'][0, 0]
        elif self.fom == 's8':
            f = mdir['fish_all'][1, 1]
        f = np.log10(f)
        return f

    def get_fom(self, angs):
        mark = self.get_mark_from_angles(angs)
        mdir = self.inspect_mark(mark)
        f = self.get_fom_from_inspection(mdir)
        print(angs, f, flush=True)
        return f
