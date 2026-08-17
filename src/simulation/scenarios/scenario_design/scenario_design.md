### Scenario design space:

Every device gets SOC bounds (0-1, 0.2-0.8, ..)
and a start soc within those bounds
and soc targets

for BESS this is just .5 at EOD
for EV we have 3 checkpoints: pre-commute 1, pre-commute 2, EOD
within bounds

this covers all situations we could hope to model in this setup

### constraints
targets should increase over the day, decreasing targets dont make sense
so target choice boilds down to:
when is first major target and what does the ramp look like

000
001
002
011
012
022
111
112
122
222 -> 10 picks of valid target structures

the max value tells us the amount of pressure
the rest tells us how it is distributed

this can all be understood as relative to start_soc
we could even include it in the table:

0|000
0|001
0|002
0|011
0|012
0|022
0,1|111
0,1|112
0,1|122
0,1,2|222 

where control starts after |




### Scenario design

- Define user stories that reflect consumer reality but reward prediction quality
- set scenario params accordingly