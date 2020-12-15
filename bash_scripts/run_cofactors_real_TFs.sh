#!/bin/bash


#TFs=( JUND MAX SPI1 SP1 HNF4A EGR1 )
TFs=( JUND MAX SPI1 SP1 HNF4A )
#TFs=( JUND HNF4A )

for TF in "${TFs[@]}"
do
    #used for DanQ, make sure that it's fixed to prev
    #models if you need
    ./run_cofactor_tl_subsample.sh $TF 5 True
    echo "DONE WITH $TF WITH TARGET"

    ./run_cofactor_tl_subsample.sh $TF 5 False
    echo "DONE WITH $TF WITHOUT TARGET"
done
