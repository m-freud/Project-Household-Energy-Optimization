ok

we found a dataset from portugal
goal: "compare control approaches"
complexity ladder:
stupid -> linear -> waterfall -> mpc

mpc hist avg -> mpc ml (xgb/rf/ridge) vs oracle

to train ml we split into 176 vs 74  (drop the 5 fold nonsense)

base load is a bit thin (11 archetypes for training or so)
so we use extragenous data from CH

CH produces similar results so far which is good enoguh

we dont do the same for pv because in practice its approx a gauss curve with some random dips
-> use weather forecast. for tbis project, ml ist a good proxy, just beat hist avg

