#!/bin/bash

#Note that some scripts used here were originally designed for STRING
#they still can be used in this context of BM

TF_NAME=$1
ITNUMBER=$2
INCLUDE_TARGET=$3
#TRAIN_DATA_FOLDER=~/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_LOWCORBM_SUBSAMPLE_I_$INCLUDE_TARGET
#TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/for_Manu/TRAIN_DATA_LOWCORBM_SUBSAMPLE_I_$INCLUDE_TARGET
#TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_STRING_DANQ_SUBSAMPLE_I_$INCLUDE_TARGET
TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_LOWCORBM_DANQ_SUBSAMPLE_I_$INCLUDE_TARGET

#MODEL_WEIGHTS_FOLDER=~/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_LOWCORBM_SUBSAMPLE_I_$INCLUDE_TARGET
#MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/for_Manu/MODEL_WEIGHTS_LOWCORBM_SUBSAMPLE_I_$INCLUDE_TARGET
#MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_STRING_DANQ_SUBSAMPLE_I_$INCLUDE_TARGET
MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_LOWCORBM_DANQ_SUBSAMPLE_I_$INCLUDE_TARGET


#RESULTS_FOLDER=~/Unibind_project/Unibind_Oriol_new/RESULTS_LOWCORBM_SUBSAMPLE_I_$INCLUDE_TARGET
#RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/for_Manu/RESULTS_LOWCORBM_SUBSAMPLE_I_$INCLUDE_TARGET
#RESULTS_FOLDER=~/Unibind_project/Unibind_Oriol_new/RESULTS_STRING_DANQ_SUBSAMPLE_I_$INCLUDE_TARGET
RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/RESULTS_LOWCORBM_DANQ_SUBSAMPLE_I_$INCLUDE_TARGET

mkdir -p $TRAIN_DATA_FOLDER
mkdir -p $MODEL_WEIGHTS_FOLDER
mkdir -p $RESULTS_FOLDER

#Get fasta and act files for 10 binding partners of TF (checked)
#cofactors
#python Run_Cofactor_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/cofactors.pickle ../data/sequences/sequences.200bp.fa $INCLUDE_TARGET $TRAIN_DATA_FOLDER/$TF_NAME\_multi
#STRING
#python Run_Cofactor_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/string_partners.pickle ../data/sequences/sequences.200bp.fa $INCLUDE_TARGET $TRAIN_DATA_FOLDER/$TF_NAME\_multi
#LOW CORRELATED BMS
python Run_Cofactor_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/not_cor_bms.pickle ../data/sequences/sequences.200bp.fa $INCLUDE_TARGET $TRAIN_DATA_FOLDER/$TF_NAME\_multi
echo "Got fasta and act files"

for (( i=1; i<=$ITNUMBER; i++ ))
do

        mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files

        #Split the data for 10 binding partners into train/validation/test (checked)
        python split_the_dataset.py $TRAIN_DATA_FOLDER/$TF_NAME\_multi/tf_peaks_$TF_NAME\_act.pkl $i $TRAIN_DATA_FOLDER/$TF_NAME\_multi/tf_peaks_$TF_NAME\_fasta.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME.h5 True
        echo "Done splitting the data for Binding partners into train/validation/test"

        #Train the multi-model with 10 outputs (binding partners of TF) (checked)
        #DanQ
        python Run_String_Analysis_Training.py $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME.h5 $MODEL_WEIGHTS_FOLDER/$TF_NAME\_real_multimodel_weights_$i 15 99 0.003 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels.pkl
        echo "Done training the multimodel with 10 outputs"

	    #Get a new data matrix for indiv model - without peaks that were used to train multi model (checked)
        python remove_training_data.py ../data/matrices/matrix2d.ReMap+UniBind.sparse.npz  $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME.h5 ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/final_df_dropped_string_real.pkl
        echo "Got a new data matrix for an indiv model"

        mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i

        #Get fasta and act files for the individual TF model (checked)
        #python get_data_for_TF.py $TF_NAME ../data/final_df_dropped_string_real.pkl ../data/sequences/sequences.200bp.fa $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl
        python get_data_for_TF_subsample_positives_old.py $TF_NAME ../data/final_df_dropped_string_real.pkl ../data/sequences/sequences.200bp.fa 1000 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl #note it was 1000 subsampling!
        echo "Got fasta and act files for the indiv model by subsampling"

        mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files

        #Split the data for the individual model into train/validation/test (checked)
        python split_the_dataset.py $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl 1 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files/$TF_NAME\_tl.h5 True
        echo "Done splitting the data for the indiv model into train/validation/test"

        #Perform training of an individual TF model with TL weights from multi-model (with inter partners)
        #and testing (checked)
        python Run_Analysis_String_Transfer_Learning.py $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files $MODEL_WEIGHTS_FOLDER/$TF_NAME\_real_indiv_weights_$i $MODEL_WEIGHTS_FOLDER/$TF_NAME\_real_multimodel_weights_$i 10 100 0.0003 5 $RESULTS_FOLDER/$TF_NAME\_$i True False $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels.pkl
        echo "Trained the individual model with TL and done testing"

        #echo "Done with the $i iteration"

done
