set log x
set xrange [1:201]
set xlabel "Nodes"
set ylabel "Total run time (s)"
set title "Weak scaling"
set key right bottom

#set term png
#set output "weak_scaling.png"

plot "along_y.dat" u 1:2 w p lw 2 t "Box extended along y", \
     "along_z.dat" u 1:2 w p lw 2 t "Box extended along z", \
     "along_yz.dat" u 1:2 w p lw 2 t "Box extended along y and z"

