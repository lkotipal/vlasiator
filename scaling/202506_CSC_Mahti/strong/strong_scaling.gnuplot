set log x
set xrange [1:201]
set log y
set xlabel "Nodes"
set ylabel "Total run time (s)"
set title "Strong scaling"
set key right bottom

#set term png
#set output "strong_scaling.png"

plot "all.dat" u 1:2 w lp lw 2 t "Total run time (s)", \
     t=0 "all.dat" u 1:(t==0?y0=$1*$2:y0, t=t+1, y0/$1) w lp lw 2 t "Ideal scaling"

