#!/bin/bash
#PBS -q k20
#PBS -l nodes=1:ppn=2
#PBS -l walltime=48:00:00
#PBS -N mallocMC_Bench_mMc_log

## ENVIRONMENT ############################
. /opt/modules-3.2.6/Modules/3.2.6/init/bash
export MODULES_NO_OUTPUT=1
module load ~/own.modules
export -n MODULES_NO_OUTPUT
uname -a
echo " "
cd ~/benchmark_in_progress

#executable: benchmark device iterations
./create_benchmarks.sh benchmark_1_mallocMC_log 2 10


