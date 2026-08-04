Critical part:
feature engineering

also:
- welche features sind wichtig?
- zeitreihen entsprechend aufbereiten
- modelle trainieren
- modelle einstöpseln
- test

hopefully we still get full hit rates without conf interval

dann doc -> feierabend


---
ev status:
we have a nice implementation almost ready.
last remeining question: what features do we use exactly?

this is best found out by training actual models

so next step:
train a model. (in isolation)
this means:

features[???] -> prediction for next 96 timesteps
improve over time