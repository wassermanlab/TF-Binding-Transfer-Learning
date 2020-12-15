#!/bin/bash

#Note that some scripts used here were originally designed for STRING
#they still can be used in this context of BM

TF_NAME=$1
BM=$2
PARTNERS=$3
ITNUMBER=$4
INCLUDE_TARGET=$5
REAL=$6
REMOVE_DATA=$7

TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_BM_LE_$REAL\_I_$INCLUDE_TARGET\_$REMOVE_DATA
MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_LE_R_$REAL\_I_$INCLUDE_TARGET\_$REMOVE_DATA
RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/RESULTS_BM_LE_$REAL\_I_$INCLUDE_TARGET\_$REMOVE_DATA

mkdir -p $TRAIN_DATA_FOLDER
mkdir -p $MODEL_WEIGHTS_FOLDER
mkdir -p $RESULTS_FOLDER

for (( i=1; i<=$ITNUMBER; i++ ))
do
    python Run_BM_Analysis_LE.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz $PARTNERS ../data/clusters_multi_modes_sorted.pickle ../data/tf_clust_corr.pickle ../data/sequences/sequences.200bp.fa $BM $REAL $INCLUDE_TARGET $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
    #python Run_BM_Analysis_LE_MEF2A.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz $PARTNERS ../data/clusters_multi_modes_sorted.pickle ../data/tf_clust_corr.pickle ../data/sequences/sequences.200bp.fa $BM $REAL $INCLUDE_TARGET $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
    echo "Got fasta and act files"
    
    mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files
    
    #Split the data for 5 binding partners into train/validation/test (checked)
    python split_the_dataset.py $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/tf_peaks_$TF_NAME\_act.pkl $i $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/tf_peaks_$TF_NAME\_fasta.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME.h5 True
    echo "Done splitting the data for Binding partners into train/validation/test"
    
    #Train the multi-model with 5 outputs (binding partners of TF) (checked)
    python Run_String_Analysis_Training.py $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME.h5 $MODEL_WEIGHTS_FOLDER/$TF_NAME\_real_multimodel_weights_$i 15 99 0.003 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels.pkl
    echo "Done training the multimodel with 5 outputs"
    
    ###############################################################################################
    ###############################################################################################
    if [ $REMOVE_DATA = "True" ]
    then
    echo "STARTED REMOVE PROCES"
    #Get a new data matrix for indiv model - without peaks that were used to train multi model (checked)
    python remove_training_data.py ../data/matrices/matrix2d.ReMap+UniBind.sparse.npz  $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME.h5 ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/final_df_dropped_string_real.pkl
    echo "Got a new data matrix for an indiv model"
    
    mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i
    
    #Get fasta and act files for the individual TF model (checked)
    python get_data_for_TF.py $TF_NAME ../data/final_df_dropped_string_real.pkl ../data/sequences/sequences.200bp.fa $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl
    echo "Got fasta and act files for the indiv model"
    
    else
    echo "NO REMOVAL PROCESS "
    mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i
    
    #Get fasta and act files for the individual TF model (checked)
    #actual_matrix.pkl is build on the base of partial matrix
    python get_data_for_TF.py $TF_NAME ../data/actual_matrix.pkl ../data/sequences/sequences.200bp.fa $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl
    echo "Got fasta and act files for the indiv model"
    
    fi
    ###############################################################################################
    ###############################################################################################
    
    mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files
    
    #Split the data for the individual model into train/validation/test (checked)
    python split_the_dataset.py $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl 1 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files/$TF_NAME\_tl.h5 True
    echo "Done splitting the data for the indiv model into train/validation/test"
    
    #Perform training of an individual TF model with TL weights from multi-model (with inter partners)
    #and testing (checked)
    python Run_Analysis_String_Transfer_Learning.py $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files $MODEL_WEIGHTS_FOLDER/$TF_NAME\_real_indiv_weights_$i $MODEL_WEIGHTS_FOLDER/$TF_NAME\_real_multimodel_weights_$i 10 100 0.0003 $PARTNERS $RESULTS_FOLDER/$TF_NAME\_$i True True $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels.pkl
    echo "Trained the individual model with TL and done testing"
    

done