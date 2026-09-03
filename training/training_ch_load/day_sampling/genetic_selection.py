'''
Genetically select best days to train models on, based on performance on the inner test set of the original data.

procedure:

- randomly create 100 sets without overlap/repetition
- score on inner test set
- keep best 50
- create 50 new sets with crossover + mutation
repeat

start with 10% mutation

create a csv file with worst, best and avg rsme performance for each cycle, call it rsme_evolution_{mutation_rate}pct_mutation.csv
create a png where we see avg rsme per cycle, name it the same way _.png
create a json with all the metadata, + best 3 sets, name it the same way _.json


we score by simulatin

'''