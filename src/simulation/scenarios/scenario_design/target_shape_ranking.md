here we rank target shapes by how interesting they are fr this project,
meaning how interesting they are in terms of control/prediction senstivity
-> if a scenario cares little about control/prediction quality, it is boring

0|000 - zero-stress baseline
0|001 - flexible, low pressure
0|002 - flexible, high pressure
0|011 - less flexible low pressure   NEIN
0|012 - smooth ramp
0|022 - less flexible high pressure
0|111 - no flexibility    NEIN
1|111 - zero-stress baseline
0|112 - like 001 but more overhead
1|112
0|122
1|122
0|222
1|222
2|222 - zero-stress baseline


any 0|1 or 0|2 is boring. charging in the morning is trivial
also starting with 1 or 2 just takes work away from the controller -> less flexibility

this means we can just start all scearios with 0|0

that leaves us with
00
01
02
12
22

for the rest
and then vary flexibility for 00 and 02