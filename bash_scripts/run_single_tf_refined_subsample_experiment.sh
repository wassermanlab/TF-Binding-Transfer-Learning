#!/bin/bash

#script to test the variability in performance
#for small data sizes
TFs=( JUND MAX SPI1 SP1 HNF4A )

TRAIN_DATA_FOLDER=~/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_INDIV_TFS_REFINED_SUBSAMPLE_50
MODEL_WEIGHTS_FOLDER=~/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_INDIV_TFS_REFINED_SUBSAMPLE_50
RESULTS_FOLDER=~/Unibind_project/Unibind_Oriol_new/RESULTS_INDIV_TFS_REFINED_SUBSAMPLE_50
REVERSE_COMPLEMENT=True

mkdir $TRAIN_DATA_FOLDER
mkdir $MODEL_WEIGHTS_FOLDER
mkdir $RESULTS_FOLDER

python split_the_dataset.py ../data/tf_peaks_50_noNs_partial.pkl 1 ../data/fasta_sequences_50_partial.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/tf_peaks_50_partial_multimodel.h5 $REVERSE_COMPLEMENT
echo "Done splitting the multimodel train data"

#train the multimodel (with 50 outputs)
python Run_Analysis_Training.py $TRAIN_DATA_FOLDER/tf_peaks_50_partial_multimodel.h5 $MODEL_WEIGHTS_FOLDER/multimodel_weights 15 99 0.003
echo "Done training the multimodel with 50 outputs"

#Get a new data matrix for indiv model - without peaks that were used to train multi model
python remove_training_data.py ../data/matrices/matrix2d.ReMap+UniBind.sparse.npz  $TRAIN_DATA_FOLDER/tf_peaks_50_partial_multimodel.h5 ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/final_df_dropped.pkl
echo "Got a new data matrix"

for TF in "${TFs[@]}"
do
        mkdir $TRAIN_DATA_FOLDER/$TF
        mkdir $RESULTS_FOLDER/$TF
        mkdir $MODEL_WEIGHTS_FOLDER/$TF
        mkdir $TRAIN_DATA_FOLDER/$TF/h5_files
        mkdir $TRAIN_DATA_FOLDER/$TF/h5_files_test
        for (( i=1; i<=5; i++ ))
        do
                python get_data_for_TF_subsample_positives.py $TF ../data/final_df_dropped.pkl ../data/sequences/sequences.200bp.fa 50 5000 $TRAIN_DATA_FOLDER/$TF/$TF\_fasta_$i.pkl $TRAIN_DATA_FOLDER/$TF/$TF\_act_$i.pkl $TRAIN_DATA_FOLDER/$TF/$TF\_test_fasta_$i.pkl $TRAIN_DATA_FOLDER/$TF/$TF\_test_act_$i.pkl
                for (( j=1; j<=5; j++ ))
                do
                        mkdir $TRAIN_DATA_FOLDER/$TF/h5_files/$TF\_$i\_$j
                        python split_the_dataset.py $TRAIN_DATA_FOLDER/$TF/$TF\_act_$i.pkl $j $TRAIN_DATA_FOLDER/$TF/$TF\_fasta_$i.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF/h5_files/$TF\_$i\_$j/$TF\_$i\_$j.h5 $REVERSE_COMPLEMENT
                        
                        mkdir $TRAIN_DATA_FOLDER/$TF/h5_files_test/$TF\_$i\_$j
                        python split_the_dataset.py $TRAIN_DATA_FOLDER/$TF/$TF\_test_act_$i.pkl $j $TRAIN_DATA_FOLDER/$TF/$TF\_test_fasta_$i.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF/h5_files_test/$TF\_$i\_$j/$TF\_$i\_$j.h5 False
                        
                         #train indiv TF models with TL  
                        mkdir $RESULTS_FOLDER/$TF/iterat_TL_$i\_$j
                        python Run_Analysis_Transfer_Learning_Subsampling.py $TRAIN_DATA_FOLDER/$TF/h5_files/$TF\_$i\_$j $TRAIN_DATA_FOLDER/$TF/h5_files_test/$TF\_$i\_$j $MODEL_WEIGHTS_FOLDER/$TF/indiv_weights_TL_$i\_$j $MODEL_WEIGHTS_FOLDER/multimodel_weights 10 100 0.0003 50 $RESULTS_FOLDER/$TF/iterat_TL_$i\_$j True 
                        echo "Trained the individual model with TL and done testing"

                        #train indiv TF models without TL (CHECK)
                        mkdir $RESULTS_FOLDER/$TF/iterat_noTL_$i\_$j
                        python Run_Analysis_Transfer_Learning_Subsampling.py $TRAIN_DATA_FOLDER/$TF/h5_files/$TF\_$i\_$j $TRAIN_DATA_FOLDER/$TF/h5_files_test/$TF\_$i\_$j $MODEL_WEIGHTS_FOLDER/$TF/indiv_weights_NoTL_$i\_$j $MODEL_WEIGHTS_FOLDER/multimodel_weights 10 100 0.003 50 $RESULTS_FOLDER/$TF/iterat_noTL_$i\_$j False 
                        echo "Trained the individual model without TL and done testing"
                done
                echo "Done with subsample iteration $i"
        done

        echo "Done with TF $TF"
done

                        
