ok careful now


we need a clear separation of responsibilities!


Simulation:
highest level.

creates households with components
tells the household to to stuff

loads the current states of each component
(load, gen, is_at_home, ...)

the components are blind!
my car does not contain today's load profile!
only the simulation knows!


runs a day for a household, with a strategy.
stores the runs somewhere (influx)


Household

gets filled with components by the simulation

gets run by the simulation
sets the decision variables for all controllables


Components

load, gen, charge



