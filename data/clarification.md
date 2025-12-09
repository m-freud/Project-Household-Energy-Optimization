# Tables

## General Information

### CPT
Price class of the user.
Determines:
- Max consumption
- Price


### ESS
Technical data of the batteries
capacity kWh
charge, discharge are max. values
efficiency applies when charging. same factor for both directions.
init, final charge values: wrong units. initial charge matters.


### Premium Charger EDP
EDP is name of the company that sells the charger


### Regular / Premium EV models
similar to ESS.
cap, charge, discharge, eff, 
consumption per km - exogen probably. we have no control over vehicle use.


## MOBIe
public charger data. 
each player has a public charger id assigned and we have known time windows where the car is at the charger.

## EVs
first table just maps players to id, we combined it with the has_ess and has_pv table.

### EV1 - first vehicle
model, .... consumption we know from the previous table about EV models.

battery levels

departure / arrival - these are fixed and matter because they set the window where we can control the EV battery.

distance / consumption - proportional. we care about consumption

trip durations - ev drives to and back from work.
this shows in the ev at home/station tables.

charger type - home and public.
power - this can only mean capacity. based on values 180, 160, 43 etc. so its kWh again.
price at PCS - ok

### EV2
analog.

### Load
consumption of the household without PV etc. just usage

### PV
PV energy. can be easily combined with Load to get net consumption

### BESS
assigns each player and the technical data for the BESS, as seen in the ess table, but by player.

### Buy Price/ Sell Price
makes sense

### Limits
we prob only care about the New CP level it sets the energy price.
since we already have energy price we may not even need this.



### Rest
macht sinn.











