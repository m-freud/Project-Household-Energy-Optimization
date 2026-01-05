
    # distribute suprlus first
    if net_load < 0:
        surplus = -net_load

        # 1. charge evs first
        evs_at_home = []
        if household.ev1 and household.ev1.at_home:
            evs_at_home.append('ev1')
        if household.ev2 and household.ev2.at_home:
            evs_at_home.append('ev2')

        for ev in evs_at_home:
            if ev.soc < req[ev]["soc"] * ev.capacity:
                ev_power_available = surplus / len(evs_at_home)
                ev_max_charge_power = min(
                    ev.max_charge, # max charge rate if you need more than one step
                    (ev.capacity - ev.soc) * 4 / ev.efficiency, # this is what is needed to get to 100% in one step.
                    ev_power_available * 4 # distribute surplus evenly. 2kWh ? you can charge 8 kW for 15min
                )
                controls[f"{ev}_power"] = ev_max_charge_power
                surplus -= ev_max_charge_power

        # 2. charge bess if any surplus left
        if surplus > 0 and household.bess:  
            bess = household.bess
            charge_power = min(surplus, bess.max_charge)
            controls["bess_power"] = charge_power  # positive for charging

    

    return controls