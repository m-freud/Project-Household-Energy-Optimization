# 1. Motivation / Overview

HEMS (House EMS) is a rapidly growing industry because hardware is getting cheaper while consumers are more aware about their energy consumption and how to lower it.
a network of HEMS (HEM Systems) gives real people autonomy so they are independent of big energy

plus its easier to get clean sustainable energy ([Greta Thunberg](https://de.wikipedia.org/wiki/Greta_Thunberg))

so  optimizing energy flows between devices has never been more relevant

two main challenges:
managing energy flows as best as possible with the data/information we have about the system

and

gathering best possible information.
this means gathering current data, and more crucially, predicting future Inputs or behaviour of devices or consumers

this project is an exercise in 
- simulating a set of 250 households with multiple devices
- comparing a set of control strategies over the population, under different scenarios

we show that employing MPC with a basic machine learning predictor can significantly lower cost over the day, as controllers are able to make better decisions.


# A. Setup and Usage

- starting point: this repo with raw data sources for everything
- run ingestion script -> orig. DB, runtime DB stuffens
- run simulation script of your choice (e.g. minimal run of 10 hh, def. scenario, mpc_oracle, mpc_hist_avg und mpc_single_step)
- inspect results in dashboard


# B. Dashboard

- pip install streamlit
- general view, single view, benchmarks, prediction explorer


# 2. System design

We simulate a set of 250 independent households.
Each household has a base load profile over the day from the consumers in the household

all households get two electric vehicles: one for commuting to and
from work (ev1) and one for domestic use (ev2). 

### EV1
EV1 have exactly one commute to and from work over the day, within time constraints (allowed windows for both commutes)

For every ev1 there is a charging station available at the job site, with a constant price that may be higher or lower than charging price from the domestic grid.


### EV2
EV2 have at least 1 commute, or maybe more

there are no charging stations for EV2 so it can only charge at home for the available rate.


### PV
200 out of 250 households get PV panels.
They generate electricity depending on current sunlight.


### BESS
150 out of the 200 PV households have an energy storage system (BESS) or battery that can be used to store energy and release it where it makes sense 

this could mean "buy cheap, sell high"
or "load surplus PV gen so it doesnt go to waste"
or simply "avoid peak loads"


### Energy flows
People living in the house consume base load. So this has to be bought from the grid at grid price
EVs can be charged, and lose their charge when driving. So far we dont implement E2H or E2G but we could easily. We just dont want to because we would have to make up technical data. Interesting xperiment though. future work.
EVs can be charged from the household or at the charging station

PV adds pv_gen so it feeds energy into the household.

BESS can store and release energy, both with a device-specific loss (e.g. 5%)


## System data

The whole setup is based on [this paper](https://www.sciencedirect.com/science/article/pii/S2352340923003372?via%3Dihub) from june 2023.

It provides an excellent playground for simluating a population of consumer / prosumer households, which it is specifically designed for.

While it is primarily intended to represent an interconnected community of households, it is a really nice basis for energy management on an individual level.
Why? 
We get realistic data of real devices and dont have to make up everything from scratch.

Included data domain:

- load profiles
- pv profiles
- ev availability / usage profiles and technical data of real EVs
- public charger technical data
- buy prices home/station
- sell prices home
- bess technical data
- ...


### inserting foreign data
It has one main weakness:
load, pv, ev profiles are somewhat artificial and lack variance.

we only have 20 load profiles, that are scaled around and repeated. they also have been somewhat normalized so they lack
the chaos of real world data

pv profiles are also limited.

ev availabilities are complely random


these issues might be irrelevant for other projects.
This being a demonstration of predictive ML, it makes sense to introduce other data.
Why?
We could of course make predictions for all of these MFs, but the question arises how to train the models.
The available data lacks variance as we only have 20 load profiles.
we could train models on foreign data, but we would get a domain disparity, which would make it hard to assess model quality isolated from this domain change
ofc we can try to predict our somewhat artificial profiles with our foreign trained models but its not th main objective here.

so we replace all our prediction targets.

Sources:

Load: load profiles from [Switzerland](https://zenodo.org/records/21398769?preview_file=dataset.zip)

PV: pv profiles from [southern germany](https://data.open-power-system-data.org/household_data)

EV availability and usage: EV usage data from [Australia](https://data.mendeley.com/datasets/gphwn7sy5n/1)

additionally, we changed ev2 from being a work vehicle with 2 commutes over the day to being a privately used car with an arbitrary number of commutes, and no access to charging stations.

this way we create a wider range of behaviors. private use is at least as relevant as commuting to and from work.


All of the foreign data sources provide much more data than necessary, so we use one subset each for runtime simulation, while withholding a larger, independent dataset for training predictive ML models.


## Scenarios

we run our simulation across a set of different scenarios meant to cover a representative set of real world application situations

scenarios are defined by start SOCs for EVs, and SOC targets over the day
so depending on the scenario we start out with more or less charge in the EVs and need to charge them more or less, at different points in time.
maybe the EVs are already close to their target charges from the start, maybe they are quite low and need to charge rapidy to ensure that our end consumers reach their destinations or jsut have enoigh reserves in the tank.


## Controllers

When running the simulation over the command line, we have the following parameters:
- household ids
- scenarios
- controllers.

arbitrary control strategies can be implemented over a controller class.

Controllers in the scope of thi project range from future-agnostic rule based systems (waterfall) to model predictive conrollers (MPC) that solve energy management as a CSP based on some prediction algorithm
predciton algorithms range from naive hist. avg. persistence to inference from ML models (XGBoost) trained on the related data.

profile predictions are single-step or predict across multiple horizons (multi-step vs multi-model)
so far, single step remains unbeaten, but time will tell.


The main purpose of this project can be seen as the comparison of different controllers/ controlling approaches.


## Simulation

Per household, we simulate one full day of 96 15-min steps.

So per step, we collect the current data of the whole system anf feed it into a controller that may internally use it to make a prediction. The controller outputs a set of control actions:
- charge/discharge BESS
- charge EVs

The simulation now uses these control actions as well as the environmental/input data to calculate the next state of the system.

rinse and repeat


### System Design Summary
So we have a whole set of simulation runs where we can evaluate controllers over a wide range of households (each with their own device arrangements, technical data and consumer/prosumer profiles) and scenarios.


# 3. Data acquisition, adaptation, ingestion

Wo kommen Daten her?
Was ändern wir?
Ingestion: Wie sehen die Daten anfangs aus und wie speichern wir sie?
-> sqlite, time series, wide vs. long, lookup tables


-> at this point all the input data is ingested and available for runtime


# 4. Simulation procedure

Diagram of components.
Design decisions.
Interfaces.
Flow of runs

## 4.1 Controllers

- stupid controllers
- heuristic rule based controllers
- prediction based controllers (MPC)
    - hist avg persist
    - ML
        - single step recursion
        - multi horizon composite predictors

## 4.2 MPC

- target function, constraints, horizons, design decisions,
performance measures (what do we have to do so it runs fast enough)

# 5. Model training

- Model choice (XGB vs. RF or other)
- Model hypam tuning
- Leakage free train val test split
- Features
- Pros / Cons of single/multi horizon

. . .


# 5. Evaluation Design

- cost KPI
- SOC target requirements (e.g. 95%) and fair comparison
- how do we avoid cheating (missed targets, charge depletion)?
    -> soc targets are hard constraints. targets are only missed if predictions postpone charging at the wrong time.
    If too many misses are observed, we employ confidence intervals or time buffers for predicitons / controls


# 5. Results

- Controller full benchmarks
    - who wins on average?
    - who wins how often?
    - is ML worth it?
- How important are the predictions, resp. ?


# 6. Whatever
- ToDos
    - figure out features
    - optimize models
- Limits
    - just one day
    - mixed data from different continents, independent of each other
    - no real validation, controlled environment


- Future / ongoing work:
    - ev state as regression?
    - optimizing models
    - e2h, e2g
    - we currently train on independent households. can we train for the same household?
        ofc we have to start somewhere but the best predictions are from the same household so we can learn individual patterns
