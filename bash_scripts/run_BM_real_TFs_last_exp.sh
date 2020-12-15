#!/bin/bash

REAL=$1
#$REMOVE_DATA=$2

declare -A TFs

#TFs=( [JUND]=1 [ATF7]=18 [NFE2L1]=1 [MAX]=7 [MNT]=7 [SREBF2]=7 
#      [SPI1]=16 [ETV4]=16 [ERG]=16 [SP1]=34 [KLF9]=34 [ZNF740]=34
#      [HNF4A]=4 [NR2C2]=4 [VDR]=46 )
TFs=( [MEF2A]=29 )

#for TF in "${!TFs[@]}"; do echo "$TF - ${TFs[$TF]}"; done
for TF in "${!TFs[@]}"
do
        #REMOVE DATA
        ./run_BM_tl_last_exp_corrected_remove.sh $TF ${TFs[$TF]} 5 5 True $REAL True 
        echo "DONE WITH $TF WITH TARGET AND REMOVED DATA"

        #REMOVE DATA
        ./run_BM_tl_last_exp_corrected_remove.sh $TF ${TFs[$TF]} 5 5 False $REAL True 
        echo "DONE WITH $TF WITHOUT TARGET AND REMOVED DATA"
        
        #KEEP DATA
        #./run_BM_tl_last_exp_corrected_remove.sh $TF ${TFs[$TF]} 5 5 True $REAL False 
        #echo "DONE WITH $TF WITH TARGET AND KEPT DATA"

        #KEEP DATA
        #./run_BM_tl_last_exp_corrected_remove.sh $TF ${TFs[$TF]} 5 5 False $REAL False 
        #echo "DONE WITH $TF WITHOUT TARGET AND KEPT DATA"

done