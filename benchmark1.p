#! /usr/bin/env gnuplot
clear
reset
#set key opaque
set key Left top left

set title "Alloc"
set ylabel "cycles"
set format y "%.1t · 10^{%T}"
#set xtics 910,50
set grid ytics xtics
set xlabel "Threads"

# Output
set output "benchmark1.pdf"
set term pdfcairo enhanced font 'Serif,14'

plot\
"benchmark1.dat" u 1:2 t "AllocMax" w lines lw 5 lt 1 lc rgb "red",\
"benchmark1.dat" u 1:3 t "AllocMin" w lines lw 5 lt 1 lc rgb "red"

