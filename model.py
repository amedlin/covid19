
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

P0 = 25000000.0
N0 = 1.0
U0 = P0 - N0
M0 = 0.0
D0 = 0.0
K0 = 0.0

T = 180.0
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
    growthN[i] = ((alpha*U[i-1]/P[i-1])*Nint(_t - ti))*dt
    assert growthN[i] >= 0.0
    growthNint = interp1d(t, growthN, copy=False, bounds_error=False, fill_value=0.0)
    resolved = growthNint(_t - tau)
    N[i] = max(N[i-1] + growthN[i] - resolved, 0.0)
    Nint = interp1d(t, N, copy=False, bounds_error=False, fill_value=0.0)
    dD = f*resolved
    D[i] = D[i-1] + dD
    M[i] = M[i-1] + (1 - f)*resolved
    P[i] = P[i-1] - dD
    assert np.isclose(P[i] + D[i], P0)
    assert N[i] <= P[i]
    assert M[i] <= P[i]
    K[i] = s*Nint(_t - ti)
    U[i] = P[i] - N[i] - M[i]
    assert U[i] >= 0.0
    print("Day: {:.2f}, Cumulative cases: {:.2f}, Recovered cases: {:.2f}, Deaths: {:.2f}, Pop: {:.2f}".format(
        _t, N[i] + M[i] + D[i], M[i], D[i], P[i]))
# end for
assert np.isclose(P[-1] + D[-1], P0)
assert np.isclose(P[-1], U[-1] + N[-1] + M[-1])

ir = (N[-1] + M[-1] + D[-1])/P0
print("Infection rate: {}".format(ir))

# Compute number  of cases we don't know
CI = N - K

all = np.floor(np.vstack((N, K, D, M, CI)))
plt.figure(figsize=(16,9))
plt.plot(t, all.T, '--', linewidth=2.0, alpha=0.6)
leg_labels = ['Actual active cases', 'Known active cases', 'Deaths', 'Recovered', 'Case ignorance']
plt.legend(leg_labels)
plt.grid(linestyle=':', color='#80808080')
plt.show()

all[(all <= 0.0)] = np.nan
plt.figure(figsize=(16,9))
plt.semilogy(t, all.T, '--', linewidth=2.0, alpha=0.6)
plt.legend(leg_labels)
plt.grid(linestyle=':', color='#80808080')
plt.show()
