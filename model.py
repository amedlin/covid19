
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date, AutoDateLocator, DateFormatter, DayLocator
from scipy.interpolate import interp1d

import pandas as pd

# See also:
# https://medium.com/@megan.higgie/without-serious-action-australia-will-run-out-of-intensive-care-beds-between-7-and-10-april-59f83b52756e

# Initial population
P0 = 25000000.0
N0 = 1.0
U0 = P0 - N0
M0 = 0.0
D0 = 0.0
K0 = 0.0

# Based on https://ourworldindata.org/grapher/total-cases-covid-19?time=4..62&country=AUS, since as of 24 March
# this is the only source I trust which is being maintained and has downloadable data.
ourworldindata_day_0 = datetime(2020, 1, 21)
case0_known_date = date2num(datetime(2020, 1, 25))
start = case0_known_date - 20.0  # Guesstimating patient 0 was infected this many days before 25 Jan when we know K = 1.

# Time dependent growth rate fitted to data
def alpha_timedep(_t):
    if _t < 21:
        return 0.3
    elif _t < 49:
        # Early phase before community spread and travellers from countries where spread is out of control
        return 0.034
    elif _t < 49 + 20:
        # Community spread + travellers returning from overseas.
        # Travellers hypothesized to account for accelerated growth in cases during this period.
        return 0.35
    else:
        # Effects of social distancing should be kicking in. Value here is a total guess, will need some data
        # to tune to in a week or so.
        return 0.15
    # end if
# end func

# Simulation period (days)
T = 240.0
# T = 365.0
end = start + T
# Time step
dt = 0.1
# Growth coefficient
# alpha = lambda k: 0.3
alpha = alpha_timedep
# Fatality rate as proportion of total actual cases who are infected for fixed duration tau.
f = 0.03
# Proportion of infected after incubation that become symptomatic (fraction that becomes 'known')
s = 0.5
# Incubation period (period until being contagious - the period until people notice symptoms is longer)
ti = 3.5
# Confirmation period - from onset of contagion to diagnosis (includes delay from contagiousness to noticing symptoms)
tc = 3.0
# Resolution period (recovery or death), relative to time of infection. Written here as incubation period plus some days of sickness.
tau = ti + 21.0
# Herd immunity threshold (proportion of population)
h = 0.5
# Proportion of known infected requiring hospitalization
phosp = 0.1
# Number of hospital beds per 1000 people (https://en.wikipedia.org/wiki/List_of_OECD_countries_by_hospital_beds)
beds_per_thousand_AUST = 3.84
# Available beds in the system
nbeds = beds_per_thousand_AUST*(P0/1000.0)
# Available ICU beds per 100,000 people (https://www.anzics.com.au/wp-content/uploads/2019/10/2018-ANZICS-CORE-Report.pdf, p.7)
icu_beds_per_hundredthousand_AUST = 8.92
n_icu_beds = icu_beds_per_hundredthousand_AUST*(P0/100000.0)


# Initialization
L = int(T/dt)  # Total number of samples in time series.
P = np.zeros(L); P[0] = P0
N = np.zeros(L); N[0] = N0
# Growth rate, new cases per dt
G = np.zeros(L)  # ; growthN[0] = 0.0
# New known cases
new_cases = np.zeros(L)
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
    G[i] = alpha(_t)*(h*P[i-1] - N[i-1] - M[i-1])/(h*P[i-1])*Nint(_t - ti)*dt
    assert G[i] >= 0.0
    Gint = interp1d(t, G, copy=False, bounds_error=False, fill_value=0.0)
    resolved = Gint(_t - tau)
    N[i] = max(N[i-1] + G[i] - resolved, 0.0)
    Nint = interp1d(t, N, copy=False, bounds_error=False, fill_value=0.0)
    dD = f*resolved
    D[i] = D[i-1] + dD
    M[i] = M[i-1] + (1 - f)*resolved
    P[i] = P[i-1] - dD
    assert np.isclose(P[i] + D[i], P0)
    assert N[i] <= P[i]
    assert M[i] <= P[i]
    new_cases[i] = s*Gint(_t - ti - tc)  # known new infections this time interval
    K[i] = K[i-1] + new_cases[i] - s*resolved
    assert K[i] >= 0.0
    U[i] = P[i] - N[i] - M[i]
    assert U[i] >= 0.0
    # print("Day: {:.2f}, Cumulative cases: {:.2f}, Recovered cases: {:.2f}, Deaths: {:.2f}, Pop: {:.2f}".format(
    #     _t, N[i] + M[i] + D[i], M[i], D[i], P[i]))
# end for
assert np.isclose(P[-1] + D[-1], P0)
assert np.isclose(U[-1] + N[-1] + M[-1] + D[-1], P0)

ir = (N[-1] + M[-1] + D[-1])/P0
print("Infection rate: {:.3f}".format(ir))

# Compute cumulative cases (total number infected out of initial population)
Ncum = N + M + D
# Compute number of cases we don't know
CI = N - K
CI[(CI < 0)] = np.nan
# Compute number of hospital beds needed
beds = phosp*K
# Compute new known cases per dt
newC_rate = new_cases/dt
# Compute total known cases
totKnown =  np.cumsum(new_cases)

# Load Australia real data. Source from https://covid.ourworldindata.org/data/total_cases.csv,
# https://covid.ourworldindata.org/data/total_deaths.csv
# case_col = 'Australia'
# cases_real = pd.read_csv('total_world_cases-covid-19-who.csv', usecols=['date', 'Australia'], parse_dates=['date'])
# cases_real['Date'] = cases_real['date'].transform(lambda d: date2num(d.date()))
# deaths_real = pd.read_csv('total_world_deaths-covid-19-who.csv', usecols=['date', 'Australia'], parse_dates=['date'])
# deaths_real['Date'] = deaths_real['date'].transform(lambda d: date2num(d.date()))
country = 'Australia'
case_col = 'Total Known'
cases_real = pd.read_csv('total-cases-covid-19.csv')
cases_real = cases_real[(cases_real['Entity'] == country)]
cases_real.rename(columns={'Total confirmed cases of COVID-19 (cases)': case_col}, inplace=True)
cases_real['Date'] = cases_real['Year'].transform(lambda d: date2num((ourworldindata_day_0 + timedelta(days=d)).date()))
deaths_real = pd.read_csv('total-deaths-covid-19.csv')
deaths_real = deaths_real[(deaths_real['Entity'] == country)]
deaths_real.rename(columns={'Total confirmed deaths due to COVID-19 (deaths)': case_col}, inplace=True)
deaths_real['Date'] = deaths_real['Year'].transform(lambda d: date2num((ourworldindata_day_0 + timedelta(days=d)).date()))

# Plot
locator = AutoDateLocator()
formatter = DateFormatter('%d %b %Y')
day_locator = DayLocator()

all = np.floor(np.vstack((Ncum, N, totKnown, K, newC_rate, D, M, CI, beds)))

# Plot linear scale
plt.figure(figsize=(16,9))
plt.plot(start + t, all.T/1.0e6, linewidth=2.0, alpha=0.6)
leg_labels = ['Total actual infections', 'Actual active cases', 'Total known cases', 'Known active cases',
              'Case discovery rate', 'Total deaths', 'Recovered', 'Case ignorance', 'Hospital beds needed']
plt.legend(leg_labels)
plt.grid(linestyle=':', color='#80808080')
plt.gca().axhline(nbeds/1.0e6, color='#808080', linestyle='--')
plt.text(start + 10.0, nbeds*1.01/1.0e6, 'AU Hospital total capacity (beds)', va='bottom')
now = datetime.now()
today_str = now.strftime('%d/%m/%Y')
today = date2num(now)
plt.gca().axvline(today, color='#808080', linestyle='--')
plt.text(today + 0.25, nbeds*1.01/1.0e6, 'Today ' + today_str, va='bottom')
# plt.ylim(None, P0)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().tick_params(axis='y', right=True, labelright=True, which='both')
plt.xlim(start, end)
plt.ylim(0, None)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number # (millions of people or beds)', fontsize=14)
plt.title('Modelling COVID-19 epidemic lifecycle in Australia', fontsize=16)
# Add plot of population still available to function
ax2 = plt.twinx()
ax2.plot(start + t, 1.0 - s*N/P, '-.', linewidth=1, alpha=0.8)
ax2.set_ylim(0, None)
ax2.set_ylabel('Capacitant population fraction (dash-dot line)', fontsize=14)
# plt.savefig('Linear_scale_covid_growth_Australia_forecast_' + now.strftime('%Y%m%d') + '.png', dpi=300)
plt.show()

# Plot log scale
all[(all <= 0.0)] = np.nan
plt.figure(figsize=(16,9))
plt.semilogy(start + t, all.T, linewidth=2.0, alpha=0.6)
plt.semilogy(cases_real['Date'], cases_real[case_col], 'x', color='C2', alpha=0.6)
plt.semilogy(deaths_real['Date'], deaths_real[case_col], 'x', color='C5', alpha=0.6)
plt.legend(leg_labels + ['AU cases', 'AU deaths'])
plt.grid(linestyle=':', color='#80808080')
plt.grid(linestyle=':', color='#90909080', axis='both', which='minor', alpha=0.2)
plt.gca().axhline(nbeds, color='#808080', linestyle='--')
plt.text(start + 10.0, nbeds*1.05, 'AU Hospital total capacity (beds)', va='bottom')
plt.gca().axhline(n_icu_beds, color='#808080', linestyle='--')
plt.text(start + 10.0, n_icu_beds*1.05, 'AU Hospital ICU capacity (beds)', va='bottom')
plt.gca().axvline(today, color='#808080', linestyle='--')
plt.text(today + 0.25, nbeds*1.05, 'Today ' + today_str, va='bottom')
# plt.ylim(None, P0)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().xaxis.set_minor_locator(day_locator)
plt.gca().tick_params(axis='y', right=True, labelright=True, which='both')
plt.gca().tick_params(axis='x', top=True, which='both')
plt.xlim(start, end)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number # (people or beds, LOG scale)', fontsize=14)
plt.title('Modelling COVID-19 epidemic lifecycle (LOG scale)', fontsize=16)
# plt.savefig('Log_scale_covid_growth_Australia_forecast_' + now.strftime('%Y%m%d') + '.png', dpi=300)
plt.show()

# Plot ratios
plt.figure(figsize=(16,9))
plt.semilogy(start + t, np.vstack((N/K, Ncum/D, D/totKnown, N/newC_rate)).T, linewidth=2.0, alpha=0.6)
leg_labels = ['Actual active cases/Known active cases', 'Total actual infections/Total deaths',
              'Total deaths/Total known cases (mortality)', 'Actual active cases/New case rate']
plt.legend(leg_labels)
plt.grid(linestyle=':', color='#80808080')
plt.grid(linestyle=':', color='#90909080', axis='both', which='minor', alpha=0.2)
# today = date2num(datetime.now())
# plt.gca().axvline(today, color='#808080', linestyle='--')
# plt.text(today + 0.25, 10, 'Today', va='bottom')
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().tick_params(axis='y', right=True, labelright=True, which='both')
plt.gca().axvline(today, color='#808080', linestyle='--')
plt.text(today + 0.25, 1, 'Today ' + today_str, va='bottom')
plt.xlim(case0_known_date, end)
# plt.ylim(1, 1e4)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Ratio', fontsize=14)
plt.title('Modelling COVID-19: Reckoning ratios', fontsize=16)
# plt.savefig('Reckoning_ratios_Australia_forecast_' + now.strftime('%Y%m%d') + '.png', dpi=300)
plt.show()
