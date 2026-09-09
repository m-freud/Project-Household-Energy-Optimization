# Project Household Energy Optimization

## what am i looking at
this is my attempt to compare different control algorithms for a virtual home energy management system (HEMS)

based on a [paper](./data/paper.pdf) i found online, and the attached [spreadsheet](./data/energy_community_data.xlsx)


### data description
The dataset represents a community of 250 residential households with 2 electric vehicles.
200 houses get PV as well
150 households get a battery (BESS)

We are provided with full-day profiles for all households, including:

- load (=base_load): energy consumption by house inhabitants
- pv: pv generation (some houses have pv, some dont)
- ev profiles for 2 vehicles:
    - ev_at_home (1 or 0)
    - ev_at_station (1 or 0)


a profile is a time series of 96 values (1 per 15 minutes) amounting to one full day