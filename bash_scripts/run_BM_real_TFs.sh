#!/bin/bash

REAL=$1

declare -A TFs

TFs=( [JUND]=1 [MAX]=7 [SPI1]=16 [SP1]=34 [HNF4A]=4 )
#TFs=( [EGR1]=34 )
#TFs=( [JUND]=1 [HNF4A]=4 )
#TFs=( JUND MAX SPI1 SP1 HNF4A )


#for TF in "${!TFs[@]}"; do echo "$TF - ${TFs[$TF]}"; done
for TF in "${!TFs[@]}"
do

	#./run_BM_tl_subsample.sh $TF ${TFs[$TF]} 5 5 True $REAL
    #    echo "DONE WITH $TF WITH TARGET"
    #DanQ
    ./run_BM_tl_subsample_DanQ.sh $TF ${TFs[$TF]} 5 5 True $REAL
        echo "DONE WITH $TF WITH TARGET"

	#./run_BM_tl_subsample.sh $TF ${TFs[$TF]} 5 5 False $REAL
    #    echo "DONE WITH $TF WITHOUT TARGET"
    #DanQ
    ./run_BM_tl_subsample_DanQ.sh $TF ${TFs[$TF]} 5 5 False $REAL
        echo "DONE WITH $TF WITHOUT TARGET"
    
done
