# Scaling results obtained on CSC's Mahti CPU nodes in June 2025

Runs done on dev at fd9292d6.

Scaling runs done with a 3D IPshock setup slightly modified from the one given in samples/IPshock.

Data for the logfile's "Total run time" as well as Phiprof's "Propagate", "Spatial-space", and "Velocity-space" timers are included in the dat files in the subfolders.

Plot the results with `gnuplot -p <script>.gnuplot` for png plots, comment the relevant lines to obtain interactive plots instead.

# Weak scaling

Test run on 1 node as in the given cfg file, and extended on *n* nodes by expanding the domain by a factor *n* along the *y* and *z* dimensions respectively, and both along *y* and *z* for the cases where the node count is *m^2*. All runs using 16 tasks per node, 16 threads per task (multihtreading).

![](weak/weak_scaling.png)
![](weak/weak_scaling_efficiency.png)

# Strong scaling

Two cases run, "light" on 2 to 200 nodes, "medium" with 4x larger box (extended in *y* and *z* and lower phase-space density threshold)

Tests run with the given cfg. All runs using 16 tasks per node, 16 threads per task (multihtreading). Change variable `case` in gnuplot scripts between "light" and "medium" to plot the different cases.

![](strong/strong_scaling.png)
![](strong/strong_scaling_efficiency.png)
