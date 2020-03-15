
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

P0 = 10000.0
N0 = 1.0
U0 = P0 - N0
M0 = 0.0
D0 = 0.0
K0 = 0.0

T = 365.0
dt = 1.0
alpha = 0.22
f = 0.01
s = 0.95
ti = 5.0
tau = 17.0

# Initialization
L = int(T/dt)
P = np.zeros(L); P[0] = P0
N = np.zeros(L); N[0] = N0
growthN = np.zeros(L)  # ; growthN[0] = 0.0
U = np.zeros(L); U[0] = U0
M = np.zeros(L); M[0] = M0
D = np.zeros(L); D[0] = D0
K = np.zeros(L); K[0] = K0
t = np.arange(L)*dt

# Make interpolators for querying time series at past times
Nint = interp1d(t, N, copy=False, bounds_error=False, fill_value=0.0)

# Updates
for i in range(1, L):
    _t = t[i]
    # N[i] = max(N[i-1] + ((alpha*(P[i-1] - M[i-1])/P[i-1])*Nint(_t - ti) - Nint(_t - tau))*dt, 0.0)
    growthN[i] = ((alpha*max(P[i-1] - N[i-1] - M[i-1], 0.0)/P[i-1])*Nint(_t - ti))*dt
    assert growthN[i] >= 0.0
    N[i] = max(N[i-1] + growthN[i] - (Nint(_t - tau) - Nint(_t - tau - dt)), 0.0)
    Nint = interp1d(t, N, copy=False, bounds_error=False, fill_value=0.0)
    growthNint = interp1d(t, growthN, copy=False, bounds_error=False, fill_value=0.0)
    dD = f*growthNint(_t - tau)
    D[i] = D[i-1] + dD
    M[i] = M[i-1] + (1 - f)*growthNint(_t - tau)
    P[i] = P[i-1] - dD
    assert N[i] <= P[i]
    assert M[i] <= P[i]
    K[i] = s*Nint(_t - ti)
    U[i] = P0 - N[i] - M[i]
    assert np.isclose(P[i] + D[i], P0)
    print("Day: {:.2f}, Cumulative cases: {:.2f}, Recovered cases: {:.2f}, Deaths: {:.2f}, Pop: {:.2f}".format(
        _t, N[i] + M[i] + D[i], M[i], D[i], P[i]))
# end for

all = np.vstack((N, K, D, M))
plt.figure(figsize=(16,9))
plt.plot(t, all.T, 'x--', linewidth=2.0, alpha=0.7)
plt.legend(['Actual active cases', 'Known active cases', 'Deaths', 'Recovered'])
plt.grid(linestyle=':', color='#80808080')
plt.show()

all[(all == 0.0)] = np.nan
plt.figure(figsize=(16,9))
plt.semilogy(t, all.T, 'x--', linewidth=2.0, alpha=0.7)
plt.legend(['Actual active cases', 'Known active cases', 'Deaths', 'Recovered'])
plt.grid(linestyle=':', color='#80808080')
plt.show()

pass
