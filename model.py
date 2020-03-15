
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date, AutoDateLocator, DateFormatter
from scipy.interpolate import interp1d

import pandas as pd


# Initial population
P0 = 25000000.0
N0 = 1.0
U0 = P0 - N0
M0 = 0.0
D0 = 0.0
K0 = 0.0
start_date = (2020, 2, 26)
start = date2num(datetime(*start_date)) - 41.0  # Guesstimating case 0 was 41 days before 26 Feb when we know K = 20, N ~ 40.

def alpha_variable(k):
    if k < 1000:
        return 0.22
    elif k < 10000:
        return 0.16
    else:
        return 0.1
    # end if
# end func

# Simulation period (days)
T = 365.0
end = start + T
# Time step
dt = 1.0
# Growth coefficient
alpha = lambda k: 0.22
# alpha = lambda k: 0.22 if k < 1000 else 0.1
# alpha = alpha_variable
# Fatality rate as proportion of those who are sick for fixed duration tau
f = 0.015
# Proportion of infected after incubation that become symptomatic
s = 0.95
# Incubation period
ti = 5.5
# Confirmation period - from onset of symptoms to diagnosis
tc = 2.0
# Resolution period (recovery or death), 17 days after symptoms noticed
tau = 12.5 + ti
# Herd immunity threshold (proportion of population)
h = 0.5
# Proportion of known infected requiring hospitalization
phosp = 0.2
# Number of hospital beds per 1000 people (https://en.wikipedia.org/wiki/List_of_OECD_countries_by_hospital_beds)
beds_per_thousand_AUST = 3.84
# Available beds in the system
nbeds = beds_per_thousand_AUST*(P0/1000.0)
print('nbeds = {}'.format(nbeds))

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
Nint = interp1d(t, N, copy=False, bounds_error=False, fill_value=N0)

# Updates
for i in range(1, L):
    _t = t[i]
    # N[i] = max(N[i-1] + ((alpha*(P[i-1] - M[i-1])/P[i-1])*Nint(_t - ti) - Nint(_t - tau))*dt, 0.0)
    growthN[i] = alpha(K[i-1])*(h*P[i-1] - N[i-1] - M[i-1])/(h*P[i-1])*Nint(_t - ti)*dt
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
    K[i] = s*Nint(_t - ti - tc)
    U[i] = P[i] - N[i] - M[i]
    assert U[i] >= 0.0
    # print("Day: {:.2f}, Cumulative cases: {:.2f}, Recovered cases: {:.2f}, Deaths: {:.2f}, Pop: {:.2f}".format(
    #     _t, N[i] + M[i] + D[i], M[i], D[i], P[i]))
# end for
assert np.isclose(P[-1] + D[-1], P0)
assert np.isclose(P[-1], U[-1] + N[-1] + M[-1])

ir = (N[-1] + M[-1] + D[-1])/P0
print("Infection rate: {}".format(ir))

# Compute cumulative cases (total number infected out of initial population)
Ncum = N + M + D
# Compute number  of cases we don't know
CI = N - K
CI[(CI < 0)] = np.nan
# Compute number of hospital beds needed
beds = phosp*K

# Load Australia real date
# data_real = pd.read_csv('total-AU-cases-covid-19-who.csv', dtype={'Country': str, 'Day Of Year': int, 'Total confirmed cases of COVID-19': float})
# case_col = 'Total confirmed cases'
# data_real.rename(columns={'Total confirmed cases of COVID-19': case_col}, inplace=True)
# data_real['Date'] = data_real['Day Of Year'].transform(lambda d: date2num(datetime.strptime('2020 ' + str(d), '%Y %j')))

case_col = 'Australia'
data_real = pd.read_csv('total_world_cases-covid-19-who.csv', usecols=['date', 'Australia'], parse_dates=['date'])
data_real['Date'] = data_real['date'].transform(lambda d: date2num(d.date()))

# Plot
locator = AutoDateLocator()
formatter = DateFormatter('%d %b %Y')

all = np.floor(np.vstack((Ncum, N, K, D, M, CI, beds)))
plt.figure(figsize=(16,9))
plt.plot(start + t, all.T, linewidth=2.0, alpha=0.6)
leg_labels = ['Total infections', 'Actual active cases', 'Known active cases', 'Deaths', 'Recovered', 'Case ignorance', 'Hospital beds needed']
plt.legend(leg_labels)
plt.grid(linestyle=':', color='#80808080')
plt.gca().axhline(nbeds, color='#808080', linestyle='--')
plt.text(start + 10.0, nbeds*1.01, 'Hospital total capacity (beds)', va='bottom')
# plt.ylim(None, P0)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().tick_params(axis='y', right=True, labelright=True, which='both')
plt.xlim(start, end)
plt.xlabel('Date')
plt.ylabel('Number # (people or beds)')
plt.title('Modelling exponential COVID-19 viral spread')
plt.show()

all[(all <= 0.0)] = np.nan
plt.figure(figsize=(16,9))
plt.semilogy(start + t, all.T, linewidth=2.0, alpha=0.6)
plt.legend(leg_labels)
plt.semilogy(data_real['Date'], data_real[case_col], 'x', color='C2')
plt.grid(linestyle=':', color='#80808080')
plt.gca().axhline(nbeds, color='#808080', linestyle='--')
plt.text(start + 10.0, nbeds*1.05, 'Hospital total capacity (beds)', va='bottom')
# plt.ylim(None, P0)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().tick_params(axis='y', right=True, labelright=True, which='both')
plt.xlim(start, end)
plt.xlabel('Date')
plt.ylabel('Number # (people or beds, LOG scale)')
plt.title('Modelling exponential COVID-19 viral spread (LOG scale)')
plt.show()
