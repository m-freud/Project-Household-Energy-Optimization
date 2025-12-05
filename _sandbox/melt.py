import pandas as pd

df = pd.DataFrame({
 "player":["A","B","C","D"],
 "t1":[10,11,12,13],
 "t2":[14,15,16,17],
 "t3":[18,19,20,21],
 "t4":[22,23,24,25],
 "t5":[26,27,28,29],
 "t6":[30,31,32,33],
 "t7":[34,35,36,37],
})

# print(df)

# melt into time series format
df_melted = df.melt(id_vars="player", var_name="time", value_name="value").sort_values(by=["player","time"]).reset_index(drop=True)
print(df_melted)