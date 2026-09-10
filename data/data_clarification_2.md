# Data clarification - what data do we have and how do we use it?

## General Information
This sheet is the basis for all other tables.
Overview. Not directly used in Optimization.

### "Amount"
Player Id | has PV | has ESS

### "Total"
Players with X

### "Contractual Power Term"
=> "Limits"
Players per price class

### triple tarif
=> "Buy Price"
Power buy price based on price class, daytime bracket (on-peak, mid-peak, ...)

### ESS
=> "BESS"
ESS techn. data. Players per ESS

### Premium Charger
=> "Max Dis-Charge"
Players per home charger id/->capacity

### Regular, Premium EV models
=> "EVs", etc
tech data and player per vehicle id

### EV Depart Arrival
=> EV Buy price.
the ev usage profiles tell us the charger currently available to our EV (home, public).
We dont change the usage of the car so what matters to us is just the buy price.

## MOBIe EV chargers
=> EV Buy price
last column is the buy price used in the buy price table.

evtl capacity concerns

## EVs
by Player ID

everything EV related

Capacity, ... , efficiency -> relevant for charging
Consumption, distance -> charging baseline to get everywhere

Power (kW) = public charger capacity

buy price mentioned again


## Load
Input variable. 

## PV
Input variable.

## BESS
per player:
max charge, efficiency, initial/("final")


## Buy Price
Household buy price for when PV + ESS_+ < load

## Sell Price
analog

## Limits
per player:
basically just fix cost

## EV1 Buy Price
home price - drive - public price - drive - home price

## EV1 Load
by player id

power consumption of the vehicle!
we dont have to worry about km or time driven.
we see the consumption of the device. could be a toaster now (with 9X% eff.)

## Max EV1 Dis-Charge
capacities of the chargers.

per player:
charge before charger has to reload
if it has 10kWh and your car can charge 2kW but it needs 20kWh then after 5h the charger will be empty and the car will be left wanting.
