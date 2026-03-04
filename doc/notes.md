
'''
The end product is a comparison of different approaches to optimize energy consumption for a household.

How would we sell that? What does the demo look like?
- Dashboard. Pre-calculate everything. Pick household (pretty ui with mini load curve/cluster selection). Pick optimizer.
- why not streamlit? curves are generally deterministic -> no need for live simulation. tag filtering etc is just easier in a dashboard. + show SQL skills with InfluxQL
- then py nb with results

View:
[Household raw profiles: load, pv, net load]                                    [Household optimized profiles: load, pv, battery, ev, net load]
                                                                    ===>            
[Household cash profiles: buy price, sell price, net cost]                      [Household optimized cash profiles: buy price, sell price, net cost]


Portfolio (draft):

Executive Summary (1 Screen, KPIs) ⚡️4
Problem & Motivation
Data & Tools
Baseline (pre-optimization)
Analysis / Clustering
Optimization Logic
Results (€/%, Charts) 💥
Limitations & Next Steps


So our sauce needs to to two things:
- Enhance influxDB with optmized profiles so we can see them in the dashboard 
    -> script that calls every function for every household and writes back to influxDB

- provide analysis tools for the notebook

'''


