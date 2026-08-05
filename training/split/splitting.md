to train a prediction model, we need to train it on a training set of households.

eg 100:150 split

however, some of the profiles for base load, pv gen, ev status are scaled duplicates of each other (same shape with factor k)

so we need a set of heterogenous curves for training (and testing)

we dont want to recombine curves because it breaks intra household correlations

if we strictly enforce heterogenity, we end up with only 25 distinct households

if we allow duplicate ev status profiles, we get some more
and we can allow them because naturally on a 3x96 grid there will be some duplicate behaviours