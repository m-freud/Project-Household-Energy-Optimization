Hallo



This is a HEMS study

we explore various approaches to fulfilling soc targets in household networks of devices

like: house + battery + EVs + solar

thats the setup

private solar is on the rise in europe and thats good
we get more consumer independence and use clean energy for that
consumers can even export energy if they want
this is nice

we want to support these consumers 
if you get a pv and a ev and a bess you benefit a lot from smart energy management

when do i charge bess?
when do i charge EVs?
when do i export electricity?

this is a question of availability (pv) but also of current price
-> price arbitrage is fun

so to summarize we want to help people save money with their multi device setups

people who are already using it will be happier and people who arent might be more interested

now there are different approaches to managing energy
we could just "charge when cheap" but this doesnt tell us how much or when energy is cheapest
we could devise a set of rules and arrange them in a waterfall model but this relies on us encoding explicit rules so it is limited by our creativity but also the rules are not precise enough
also, we are missing future information

if we had future information, could we find a perfect control agorithm?
yes, with MPC -> control task becomes a CSP
MPC oracle is unbeatable

so now our real limit is the quality of our predictions
the simplest prediction we can make is just a presisted constant
could be last value, avg value, ...
avg value is better than last value
and performs surprisingly well vs waterfall
although we only did very simple waterfall
ofc we dont need complex waterfall, mpc is better anyway

so how do we beat mpc with hist avg?
we make smarter preditions
ML predictions

sadly our data doesnt give us weather, or weekday, or anything else besides the profiles themselves
but this makes it more interesting challenge

we only have the current state of the profile
for single days of households

u start with basically nothing

ofc we can easily train some model to predict the next state of household
(we care about load, pv, ev states)

we basically tried 3 models:
ridge (too stupid), random forest (too slow), xgb (slightly better than rf but way faster because trees are mor deliberate)

so yea you can train xgb on all prediciton targets and get decent results

but can we get better?

you see we actually have quite limited data:
its 250 households but they share only 20 load profiles with different scalings
so we have 20 load profiles
if we do a cool train/test split, which we did, we end up with 15 vs 5 or sth likethat

15 profiles are not enough to really project behaviour of consumer loads into a model
and we can prove it:
- CH models beat portugal models around half of the time
- CH models are saturated only afer like 80 days

so we get benefit from consulting external datasources

what we can now also do is really find out what matters in predicting these load profiles
fun experiment:
randomly select CH load days, train a model
and measure fun stuff like 1st step rsme but also h1,h2,h4,h8

what we find out is what horizons are the most important for the final KPI, which is net cost

so with the help of ch data we can build predictrs that are well aligned for this task
plug them into an mpc controller
and get very close to an optimal control system without even relying on weather data or whatever