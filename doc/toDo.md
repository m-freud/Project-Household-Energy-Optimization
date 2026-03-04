ok so we switched to scenario based constraint definition
this is probably still broken
fix it

check if targeted_greedy does what it should
-> new baseline
-> new base policy that others can be derived from

from there: lookahead optimization! start with price, then optimize for expected pv, then for expected load
more or less
however the situation is this:
we have now added constraints so this becomes an optimization problem.
targeted_greedy is just a special case where we just use a line
from now on we strive for the curve that minimizes cost.

-> we need a new setup. a cost function

add price sensitivity = charge wherever it is cheapest


starting point:
new policy price_aware_greedy or whatever

keeps track of: 
lowest price until deadline
highest price until deadline
compares

if at lowest price: charge 100%
if at higher price: charge pivot value (avg_needed_charge * lowest_price/highest_price) but also as much as needed

-> write policy that charges at last possible moment (add 2h buffer for cars)

