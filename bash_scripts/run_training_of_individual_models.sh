#!/bin/bash

#in tools change ConvNetDeepBNAfterRelu to ConvNetDeep
ITNUMBER=$1
TRAIN_DATA_FOLDER=~/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_50_SORTED
MODEL_WEIGHTS_FOLDER=~/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_50_SORTED
RESULTS_FOLDER=~/Unibind_project/Unibind_Oriol_new/RESULTS_50_SORTED
REVERSE_COMPLEMENT=True

#Activate conda environment
#conda activate pytorch_env

mkdir $TRAIN_DATA_FOLDER
mkdir $MODEL_WEIGHTS_FOLDER
mkdir $RESULTS_FOLDER

for (( i=1; i<=$ITNUMBER; i++ ))
do
	mkdir $TRAIN_DATA_FOLDER/iterat_$i

	#split the training data for 50 multimidel into train/validation/test (CHECK)
	python split_the_dataset.py ../data/tf_peaks_50_noNs_partial.pkl $i ../data/fasta_sequences_50_partial.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/iterat_$i/tf_peaks_50_partial.h5 $REVERSE_COMPLEMENT
	echo "Done splitting the multimodel train data"

	#train the multimodel (with 50 outputs) (CHECK)
        python Run_Analysis_Training.py $TRAIN_DATA_FOLDER/iterat_$i/tf_peaks_50_partial.h5 $MODEL_WEIGHTS_FOLDER/multimodel_weights_$i 15 99 0.003
        echo "Done training the multimodel with 50 outputs"

	#Get a new data matrix for indiv model - without peaks that were used to train multi model (CHECK)
	python remove_training_data.py ../data/matrices/matrix2d.ReMap+UniBind.sparse.npz  $TRAIN_DATA_FOLDER/iterat_$i/tf_peaks_50_partial.h5 ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/final_df_dropped.pkl
        echo "Got a new data matrix"

	#get h5 files for indiv TFs (CHECK)
	mkdir $TRAIN_DATA_FOLDER/iterat_$i/h5_files
	./get_indiv_data_for_each_TF.sh ../data/Analyzed_TFs.txt ../data/final_df_dropped.pkl $TRAIN_DATA_FOLDER/iterat_$i $TRAIN_DATA_FOLDER/iterat_$i $TRAIN_DATA_FOLDER/iterat_$i/h5_files $REVERSE_COMPLEMENT
	echo "Done with h5 files"

	#train indiv TF models with TL (CHECK)
        python Run_Analysis_Transfer_Learning.py $TRAIN_DATA_FOLDER/iterat_$i/h5_files $MODEL_WEIGHTS_FOLDER/indiv_weights_TL_$i $MODEL_WEIGHTS_FOLDER/multimodel_weights_$i 10 100 0.0003 50 $RESULTS_FOLDER/iterat_TL_$i True True
        echo "Trained the individual model with TL and done testing"

	#train indiv TF models without TL (CHECK)
        python Run_Analysis_Transfer_Learning.py $TRAIN_DATA_FOLDER/iterat_$i/h5_files $MODEL_WEIGHTS_FOLDER/indiv_weights_NoTL_$i $MODEL_WEIGHTS_FOLDER/multimodel_weights_$i 10 100 0.003 50 $RESULTS_FOLDER/iterat_noTL_$i False False
        echo "Trained the individual model without TL and done testing"

	echo "DONE WITH ITERATION $i"
done
