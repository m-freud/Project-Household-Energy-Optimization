OK 

we now want to perform some energy management on the households.

so we need to understand how the curves are related.


What degrees of freedom/control do we have? What is Management?
This is what we call DECISION VARIABLES

### Grid
- buy
- sell

### PV
- charge battery
- sell to grid
- charge car

### Battery
- charge
- discharge

### EV
- charge home
- charge station

We do not model V2G, because that would require speculative assumptions about hardware capabilities and export prices for both home and public chargers. The dataset does not provide enough information to support those assumptions, so EVs are treated as charge-only loads.

