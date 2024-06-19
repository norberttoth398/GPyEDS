# %% 
import numpy as np
import scipy.interpolate as interpolate


def draw_line(img, x0, y0, x1, y1, value = 1, pstep = None):
        mat = np.zeros_like(img)
        step_number  = int(abs(x0-x1) + abs(y0-y1)) #Number of steps
        if step_number < 2:
            step_number = 2
        else:
            pass
        step_size = 1.0/step_number #Increment size
        p = [] #Point array (you can return this and not modify the matrix in the last 2 lines)
        t = 0.0 #Step current increment
        for i in range(step_number):
            p.append([int(round(x1 * t + x0 * (1 - t))), int(round(y1 * t + y0 * (1 - t)))])
            t+=step_size
        
        p = np.asarray(p)
        mat[p[:,0],p[:, 1]] = value

        print(p)
        return img[mat.astype("bool")]

def draw_proj_box(img, x0, y0, x1, y1, phi = 45, value = 1, pstep = 1):
    v = np.asarray([x1, y1]) - np.asarray([x0,y0])
    theta = np.arccos(v[1]/(np.sqrt(v[1]**2+v[0]**2)))
    phi = theta + np.pi*(phi/180)
    #phi = theta + phi
    perp = np.asarray([np.sin(phi), np.cos(phi)])
    #perp = np.asarray([-v[1], v[0]])/(np.sqrt(v[1]**2+v[0]**2))#
    max_val = np.max(img[~np.isnan(img)])

    mat = np.zeros_like(img)
    step_number  = int(abs(x0-x1) + abs(y0-y1)) #Number of steps
    if step_number < 2:
            step_number = 2
    else:
            pass
    step_size = 1.0/step_number #Increment size
    p = [] #Point array (you can return this and not modify the matrix in the last 2 lines)
    t = 0.0 #Step current increment
    for i in range(step_number):
        p.append([int(round(x1 * t + x0 * (1 - t))), int(round(y1 * t + y0 * (1 - t)))])
        t+=step_size

    conc = []
    #create array of all steps perpendicular to vector between two end points
    nsteps = np.linspace(-pstep, pstep, 2*pstep+1)
    for item in p:
        mat = np.zeros_like(img)
        #coords perpendicular to vector
        coords = np.asarray([item+j*perp for j in nsteps], dtype = "int64")
        #get all vals
        mat[coords[:,0], coords[:,1]] = value
        vals = img[mat.astype("bool")]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
             vals = np.asarray([-10*max_val])
        else:
             pass
        #only want median to get rid of noise
        conc.append(np.median(vals))
    
    return np.asarray(conc)

def align_once(inputmatrix, pos, theta):
    #slope = theta[5]
    #intercept = theta[6]
    #ax, ay, bx, by, ww, slope, intercept = P[0:Nvalues]
    #ww = int(theta[4])
    n = np.max(pos).astype("int64")
    ax = int(theta[0])
    ay = int(theta[1])
    bx = int(theta[2])
    by = int(theta[3])
    #phi = theta[7]

    vals = draw_proj_box(inputmatrix, ax, ay, bx, by, 90, pstep = 5)
    steps = np.linspace(0, len(vals)-1, len(vals))
    steps = steps[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    if 0 in steps:
        pass
    else:
        if steps[0]-1 != 0:
            steps = np.concatenate([[0, steps[0]-1], steps])
            vals = np.concatenate([[0,0], vals])
        else:
            steps = np.concatenate([[0], steps])
            vals = np.concatenate([[0], vals])

    interpolation = interpolate.interp1d(steps, vals)
    trial_x = interpolation(np.linspace(0, len(vals)-1, n))[pos]

    return trial_x


def align(inputmatrix, theta, pos, **kwargs):
    slope = theta[5]
    intercept = theta[6]
    #ax, ay, bx, by, ww, slope, intercept = P[0:Nvalues]
    ww = int(theta[4])
    n = np.max(pos)
    #print(pos, pos.shape)
    ax = int(theta[0])
    ay = int(theta[1])
    bx = int(theta[2])
    by = int(theta[3])
    phi = theta[7]

    vals = draw_proj_box(inputmatrix, ax, ay, bx, by, phi, pstep = ww)
    steps = np.linspace(0, len(vals)-1, len(vals))
    steps = steps[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    if 0 in steps:
        pass
    else:
        if steps[0]-1 != 0:
            steps = np.concatenate([[0, steps[0]-1], steps])
            vals = np.concatenate([[0,0], vals])
        else:
            steps = np.concatenate([[0], steps])
            vals = np.concatenate([[0], vals])

    interpolation = interpolate.interp1d(steps/np.max(steps), vals)
    trial_x = interpolation(np.abs(pos/n))

    return trial_x*slope + intercept


def test_ssfun(theta, data):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    try:
        ymodel = align(xdata, len(ydata), theta)
    except IndexError:
         ymodel = np.inf
    # calc sos
    ss = sum((ymodel - ydata)**2)
    return ss

class logfuncs():
     
    def __init__(self,pmin, pmax):
          #super().__init___(pmin, pmax)
          self.pmin = pmin
          self.pmax = pmax


    def log_prior(self,theta):
        ax, ay, bx, by, ww, m, b, phi = theta
        if self.pmin[0] < ax < self.pmax[0] and self.pmin[1] < ay < self.pmax[1] and self.pmin[2] < bx < self.pmax[2] and self.pmin[3] < by < self.pmax[3] and self.pmin[4] < ww < self.pmax[4] and self.pmin[5] < m < self.pmax[5] and self.pmin[6] < b < self.pmax[6] and self.pmin[7] < phi <self.pmax[7]:
            return 0.0
        return -np.inf

    def log_likelihood(self,theta, x, y, yerr, pos):
        try:
            model = align(x, theta, pos )
        except IndexError:
            return -np.inf
        #print(model)
        sigma2 = yerr**2 + model**2
        #sigma2 = model**2
        return -0.5 * np.sum(np.divide((y - model)**2, yerr**2))
    

    def log_probability(self,theta, x, y, yerr, pos):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr, pos)
    
class simple_logfuncs():
     
    def __init__(self,pmin, pmax):
          #super().__init___(pmin, pmax)
          self.pmin = pmin
          self.pmax = pmax


    def log_prior(self,theta):
        m, b = theta
        if self.pmin[0] < m < self.pmax[0] and self.pmin[1] < b < self.pmax[1]:
            return 0.0
        return -np.inf

    def log_likelihood(self,theta, x, y, yerr):
        try:
            model = x*theta[0] + theta[1]
        except IndexError:
            return -np.inf
        sigma2 = yerr**2 + model**2
        #sigma2 = model**2
        return -0.5 * np.sum(np.divide((y - model)**2, yerr**2))
    

    def log_probability(self,theta, x, y, yerr):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)
    

def MCMC_run(x,y, uncert, params, pmin, pmax, positions, num_iter = 10000, return_ = False, name = "mcmc"):
    

    names   = ["ax","ay", "bx", "by", "ww", "m", "b", "phi"]

    funcs = logfuncs(pmin,pmax)
    from scipy.optimize import minimize
    nll = lambda *args: -1*funcs.log_likelihood(*args)
    initial = np.array(params) + 0.1 * np.random.randn(8)
    soln = minimize(nll, initial, args=(x, y, uncert, positions))
    
    print(soln.x)

    import emcee
    from multiprocessing import Pool
    pos = np.asarray(soln.x) + 1e-5 * np.random.randn(17, 8)
    nwalkers, ndim = pos.shape

    filename = str(name) + ".h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
        nwalkers, ndim, funcs.log_probability, args=(x, y, uncert, positions), backend = backend, pool = pool
        )
        sampler.run_mcmc(pos, num_iter, progress=True)
    #, moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)]
    samples = sampler.get_chain()
    import pickle
    with open("emcee_res.pkl", "wb") as f:
         pickle.dump(samples, f)

    if return_ is True:
         return sampler


def Simple_MCMC_run(x,y, uncert, params, pmin, pmax, return_ = False, name = "mcmc"):
    

    names   = ["ax","ay", "bx", "by", "ww", "m", "b", "phi"]

    funcs = simple_logfuncs(pmin,pmax)
    from scipy.optimize import minimize
    nll = lambda *args: -1*funcs.log_likelihood(*args)
    initial = np.array(params) + 0.1 * np.random.randn(2)
    soln = minimize(nll, initial, args=(x, y, uncert))
    
    print(soln.x)

    import emcee
    from multiprocessing import Pool
    pos = np.asarray(soln.x) + 1e-5 * np.random.randn(17, 2)
    nwalkers, ndim = pos.shape

    filename = str(name) + ".h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
        nwalkers, ndim, funcs.log_probability, args=(x, y, uncert), backend = backend, pool = pool
        )
        sampler.run_mcmc(pos, 10000, progress=True)
    #, moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)]
    samples = sampler.get_chain()
    import pickle
    with open("emcee_res.pkl", "wb") as f:
         pickle.dump(samples, f)

    if return_ is True:
         return sampler