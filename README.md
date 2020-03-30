# Modelling COVID-19 using coupled time-dependent differential equations

Math modelling viral epidemiology - hobby project to try to understand magnitude of unknown infections as a function of known cases.

Note that in the [ourworldindata](https://ourworldindata.org/coronavirus) site, day 0 is 21 Jan 2020. In the CSV files downloaded from this site,
some tables have the 'days' column labelled as 'Year'.

## Update 30 March 2020

I have concluded that further progress with this modelling is pointless unless I can somehow acquire reliable time series data showing the breakdown
of transmission source for Australia. The currently used data (and the fitting I performed to that data) is not valid because it is dominated by
cases imported by travellers. The model data should only be fitted to the proportion of cases which were acquired by local community transmission.
