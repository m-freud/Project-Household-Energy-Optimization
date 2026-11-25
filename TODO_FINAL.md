verify that smooth recursive interpolation (formerly anchored recursive interpolation) works as intended - bei der gelegenheit halt auch einfach den code aufräumen
-- ja und nein sieht ok aus

add some sort of prediction accuracy metric to the runs

also woran machen wir fest dass ein predictor besser ist
- next-step rmse
- plot: R(horizon), R(timestep)
- avg(R(horizon)), avg)R(timestep)


wenn wir das haben: verify that anchored is slightly more accurate than single step
on that basis: confirm that the net cost is basically the same anyway (bisheriges ergebnis)
clean up predictor files (de-spaghettify / de-slop), esp. anchored pred.
readme