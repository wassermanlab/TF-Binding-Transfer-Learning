#!/bin/bash

TF_FILE=$1
DF_FILE=$2
OUTPUT_FASTA_FOLDER=$3
OUTPUT_ACT_FOLDER=$4
OUTPUT_H5_FOLDER=$5
REVERSE_COMPLEMENT=$6

let index=0
while read line; do
  	tfs[index]="$line"
	((++index))  	
done < $TF_FILE

for tf in "${tfs[@]}"
do	
	#python script to build fasta and act (table with labels) files 
	python get_data_for_TF.py $tf $DF_FILE ../data/sequences/sequences.200bp.fa $OUTPUT_FASTA_FOLDER/$tf\_fasta.pkl $OUTPUT_ACT_FOLDER/$tf\_act.pkl

	#run the script to build the h5 file
	python split_the_dataset.py $OUTPUT_ACT_FOLDER/$tf\_act.pkl 1 $OUTPUT_FASTA_FOLDER/$tf\_fasta.pkl 0.1 0.1 $OUTPUT_H5_FOLDER/$tf.h5 $REVERSE_COMPLEMENT

	echo "Done with $tf"
	
done
