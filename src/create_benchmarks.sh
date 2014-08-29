#!/bin/bash

ITERATIONS=10

if [ "$1" != "" ] ; then
  BENCHMARK="$1"
else
  echo "no benchmarkig executable!"
  exit 1
fi

RESULT_FOLDER=result_folder_$BENCHMARK

mkdir -p ${RESULT_FOLDER}

# shortened sequence for testing
for threadcount in 16 20 24 28 32 40 48 56 64 80 96 128 160 192 224 256 320 384 448 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096 5120 6144 7168 8192 10240 12288 14336 16384; do

## full sequence
#for threadcount in 1 2 4 6 8 12 16 20 24 28 32 40 48 56 64 80 96 128 160 192 224 256 320 384 448 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096 5120 6144 7168 8192 10240 12288 14336 16384; do

  for (( i=1; i<=$ITERATIONS; i++ )) ; do
    echo "starting $threadcount threads (run $i out of ${ITERATIONS}) from $BENCHMARK"
    ./${BENCHMARK} --device=1 --threads=${threadcount} 1>/dev/zero 2>>${RESULT_FOLDER}/${BENCHMARK}_${threadcount}.txt
    echo "done"
    echo "time to break: 3s"
    sleep 3
  done

  echo "time to break: 5s"
  sleep 5
done

./result_parser ${RESULT_FOLDER}
mv output.dat ${RESULT_FOLDER}

exit 0
