
# from simulation.run_context import RunContext


# DEFAULT_HISTORY_MEASUREMENTS = [
#     "net_load",
#     "net_cost",
#     "total_consumption",
#     "total_cost",
#     "bess_soc",
#     "bess_power",
#     "ev1_soc",
#     "ev1_power",
#     "ev2_soc",
#     "ev2_power",
# ]


# _WORKER_RUN_CONTEXT: RunContext | None = None



# def _initialize_household_worker(run_context: RunContext) -> None:
#     global _WORKER_RUN_CONTEXT
#     _WORKER_RUN_CONTEXT = run_context


# def _run_household_worker(
#     player_id: int,
#     run_context: RunContext | None = None,
# ) -> dict:
#     # Worker processes create isolated DB connections; only the parent writes results.
#     from src.sqlite_connection import create_sqlite_connection

#     if run_context is None:
#         run_context = _WORKER_RUN_CONTEXT
#     if run_context is None:
#         raise RuntimeError("Household worker has no run context.")

#     worker_conn = create_sqlite_connection()
#     try:
#         sim = Simulation(
#             worker_conn,
#             ensure_results_table=False,
#             results_conn=worker_conn,
#         )
#         scenario = run_context.scenario

#         start_time = run_context.start_time
#         if start_time < 1 or start_time > sim.num_timesteps:
#             raise ValueError(f"start_time must be between 1 and {sim.num_timesteps}")

#         household = sim.create_household(player_id, run_context)
#         controller = sim.create_controller(household, run_context)
#         print(
#             f"running household {player_id} in scenario {scenario.name} "
#             f"with controller {controller.name}, run_id={run_context.run_id}",
#             flush=True,
#         )
#         for t in range(start_time, sim.num_timesteps + 1):
#             if controller.name in {
#                 "mpc_single_step",
#                 "mpc_multi_step",
#                 "mpc_composite_horizons",
#             } and (t == start_time or t % 8 == 0 or t == sim.num_timesteps):
#                 print(
#                     f"{controller.name}: household {player_id}, timestep {t}/{sim.num_timesteps}",
#                     flush=True,
#                 )
#             sim.step(household, controller, scenario, duration_hours=sim.duration_hours, time=t)

#         return {
#             "run_id": run_context.run_id,
#             "policy": controller.name,
#             "scenario": scenario.name,
#             "player_id": household.player_id,
#             "has_pv": household.has_pv,
#             "has_bess": household.has_bess,
#             "total_cost": household.total_cost,
#             "total_consumption": household.total_consumption,
#             "net_cost": sum(household.history["net_cost"].values()) * sim.duration_hours,
#             "net_load": sum(household.history["net_load"].values()) * sim.duration_hours,
#             "target_met_bess": household.has_met_final_target("bess"),
#             "target_met_ev1": household.has_met_final_target("ev1"),
#             "target_met_ev2": household.has_met_final_target("ev2"),
#             "target_met_all_bess": household.has_met_all_targets("bess"),
#             "target_met_all_ev1": household.has_met_all_targets("ev1"),
#             "target_met_all_ev2": household.has_met_all_targets("ev2"),
#             "soc_at_deadline_bess": household.soc_at_deadline("bess"),
#             "soc_at_deadline_ev1": household.soc_at_deadline("ev1"),
#             "soc_at_deadline_ev2": household.soc_at_deadline("ev2"),
#             "history": {
#                 measurement: list(household.history[measurement].items())
#                 for measurement in DEFAULT_HISTORY_MEASUREMENTS
#             },
#         }
#     finally:
#         worker_conn.close()
