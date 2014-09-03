#! /usr/bin/env gnuplot
clear
reset
#set key opaque
set key Left top right

set title "Allocation 128"
set ylabel "cycles"
set format y "%.1t · 10^{%T}"
set logscale x
set logscale y
set grid ytics xtics
set xlabel "Threads"

set style fill transparent bo

#Output
set output "benchmark1_allocation_128.pdf"
set term pdfcairo enhanced font 'Serif,14'

plot\
"benchmark1_mallocMC_128.dat" u 1:2:3 t "mallocMC" w filledcurves fs transparent lw 5 lt 1 lc rgb "red",\
"benchmark1_scatterAlloc_128.dat" u 1:2:3 t "ScatterAlloc" w filledcurves fs transparent lw 5 lt 1 lc rgb "blue",\
"benchmark1_deviceMalloc_128.dat" u 1:2:3 t "CUDA Allocator" w filledcurves fs transparent lw 5 lt 1 lc rgb "green"


set title "Free 128"
#Output
set output "benchmark1_free_128.pdf"
plot\
"benchmark1_mallocMC_128.dat" u 1:4:5 t "mallocMC" w filledcurves fs transparent lw 5 lt 1 lc rgb "red",\
"benchmark1_scatterAlloc_128.dat" u 1:4:5 t "ScatterAlloc" w filledcurves lw 5 lt 1 lc rgb "blue",\
"benchmark1_deviceMalloc_128.dat" u 1:4:5 t "CUDA Allocator" w filledcurves lw 5 lt 1 lc rgb "green"


set title "Allocation Log"
#Output
set output "benchmark1_allocation_log.pdf"
plot\
"benchmark1_mallocMC_log.dat" u 1:2:3 t "mallocMC" w filledcurves lw 5 lt 2 lc rgb "red",\
"benchmark1_scatterAlloc_log.dat" u 1:2:3 t "ScatterAlloc" w filledcurves lw 5 lt 1 lc rgb "blue"
#"benchmark1_deviceMalloc_log.dat" u 1:2:3 t "CUDA Allocator" w filledcurves lw 5 lt 1 lc rgb "green"


set title "Free log"
#Output
set output "benchmark1_free_log.pdf"
plot\
"benchmark1_mallocMC_log.dat" u 1:4:5 t "mallocMC" w filledcurves lw 5 lt 1 lc rgb "red",\
"benchmark1_scatterAlloc_log.dat" u 1:4:5 t "ScatterAlloc" w filledcurves lw 5 lt 1 lc rgb "blue"
#"benchmark1_deviceMalloc_log.dat" u 1:4:5 t "CUDA Allocator" w filledcurves lw 5 lt 1 lc rgb "green"
