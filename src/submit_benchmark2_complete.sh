#!/bin/bash
#PBS -q k20
#PBS -l nodes=kepler008:ppn=2
#PBS -l walltime=96:00:00
#PBS -N mallocMC_bench2_device_kepler008-0

## ENVIRONMENT ############################
. /opt/modules-3.2.6/Modules/3.2.6/init/bash
export MODULES_NO_OUTPUT=1
module load ~/own.modules
export -n MODULES_NO_OUTPUT
uname -a
echo " "
cd ~/benchmark_2_kepler008-0

#executable: benchmark device iterations
for i in 16 32 64 128 ; do
  ./create_benchmarks.sh benchmark_2_deviceMalloc_$i 0 3
done

