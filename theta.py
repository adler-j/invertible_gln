import scipy.io
import matplotlib.pyplot as plt
import numpy as np

n = 'theta_taylor_half' #, 'theta_taylor_single', 'theta_taylor'
mat = scipy.io.loadmat(n + '.mat')
theta = mat['theta'].squeeze()

nrm = 10.0

m = np.arange(1, len(theta) + 1)
vals = m * np.ceil(nrm / theta)
plt.semilogy(m, vals)
mstar = min(np.argmin(vals), 55)
s = int(np.ceil(nrm / mstar))

print(mstar, s)
