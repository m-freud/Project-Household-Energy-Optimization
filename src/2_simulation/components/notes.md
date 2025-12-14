components are blind.

if i create a ev class, the ev doesnt know shit about load profiles or algorithms or buy prices.
it is a car. you can drive it and you can charge it.


coordination happens in the household 

- by the household or with the household?

we can 

household_1.simulate_day(devices, optimizer)
(household.simulations gets another graph)

or we can

simulation_1 = simulator.simulate_day(household, devices, optimizer)

loader.load_to_influx(simulation_1).


we choose the first option because it is more natural.
a household has behaviour.