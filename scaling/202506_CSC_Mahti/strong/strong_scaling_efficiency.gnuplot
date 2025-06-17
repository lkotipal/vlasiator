set log x
set xrange [1:201]
set xlabel "Nodes"
set ylabel "Efficiency"
set title "Strong scaling efficiency"
set key left bottom

#set term png
#set output "strong_scaling_efficiency.png"

plot t=0 "all.dat" u 1:(t==0?y0=$2:y0, t=t+1, y0*2/($1*$2)) w lp lw 2 t "Total run time", \
     t=0 "all.dat" u 1:(t==0?y0=$3:y0, t=t+1, y0*2/($1*$3)) w lp lw 2 t "Propagate", \
     t=0 "all.dat" u 1:(t==0?y0=$4:y0, t=t+1, y0*2/($1*$4)) w lp lw 2 t "Spatial-space", \
     t=0 "all.dat" u 1:(t==0?y0=$5:y0, t=t+1, y0*2/($1*$5)) w lp lw 2 t "Velocity-space"

