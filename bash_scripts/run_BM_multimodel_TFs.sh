#!/bin/bash

declare -A TFs

TFs=( [JUND]=1 [MAX]=7 [SPI1]=16 [SP1]=34 [HNF4A]=4 )

for TF in "${!TFs[@]}"
do        
    ./run_BM_multimodel.sh $TF ${TFs[$TF]} 5
    echo "DONE WITH $TF WITH TARGET"
    
done
