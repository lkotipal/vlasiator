# Scaling results obtained on CSC's Mahti CPU nodes in June 2025

Runs done on dev at fd9292d6.

Scaling runs done with a 3D IPshock setup slightly modified from the one given in samples/IPshock.

Data for the logfile's "Total run time" as well as Phiprof's "Propagate", "Spatial-space", and "Velocity-space" timers are included in the dat files in the subfolders.

Plot the results interactively with `gnuplot -p <script>.gnuplot` for interactive plots.

Uncomment the relevant lines to obtain png output instead.

# Weak scaling

Test run on 1 node as in the given cfg file, and extended on *n* nodes by expanding the domain by a factor *n* along the *y* and *z* dimensions respectively, and both along *y* and *z* for the cases where the node count is *m^2*. All runs using 16 tasks per node, 16 threads per task (multihtreading).

# Strong scaling

Tests run with the given cfg. All runs using 16 tasks per node, 16 threads per task (multihtreading).

