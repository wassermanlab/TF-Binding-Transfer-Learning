#!/bin/bash

TF_NAME=$1
BM=$2
ITNUMBER=$3

#TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_BM_MULTIMODEL
#TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_COFACTOR_MULTIMODEL
#TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_STRING_MULTIMODEL
#TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_BESTCOR_MULTIMODEL
TRAIN_DATA_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/TRAIN_DATA_BESTSTR_MULTIMODEL

#MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_BM_MULTIMODEL
#MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_COFACTOR_MULTIMODEL
#MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_STRING_MULTIMODEL
#MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_BESTCOR_MULTIMODEL
MODEL_WEIGHTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/MODEL_WEIGHTS_BESTSTR_MULTIMODEL

#RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/RESULTS_BM_MULTIMODEL
#RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/RESULTS_COFACTOR_MULTIMODEL
#RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/RESULTS_STRING_MULTIMODEL
#RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/RESULTS_BESTCOR_MULTIMODEL
RESULTS_FOLDER=~/gnovakovskytmp/Unibind_project/Unibind_Oriol_new/RESULTS_BESTSTR_MULTIMODEL

mkdir -p $TRAIN_DATA_FOLDER
mkdir -p $MODEL_WEIGHTS_FOLDER
mkdir -p $RESULTS_FOLDER

for (( i=1; i<=$ITNUMBER; i++ ))
do
        mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
        
        #Get fasta and act files for 50 and 5 multi
        #python Run_BM_Multimodel_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/clusters_multi_modes_sorted.pickle ../data/tf_clust_corr.pickle ../data/sequences/sequences.200bp.fa $BM $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
        #cofactors
        #python Run_Cofactor_Multimodel_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/cofactors.pickle ../data/sequences/sequences.200bp.fa $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
        #STRING
        #python Run_Cofactor_Multimodel_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/string_partners.pickle ../data/sequences/sequences.200bp.fa $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
        #BESTCOR
        #python Run_Cofactor_Multimodel_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/best_cor_tfs.pickle ../data/sequences/sequences.200bp.fa $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
        #BEST STRING
        python Run_Cofactor_Multimodel_Analysis.py $TF_NAME ../data/matrices/matrix2d.ReMap+UniBind.partial.npz ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/string_partners_best.pickle ../data/sequences/sequences.200bp.fa $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i
        echo "Got fasta and act files"
        
        mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files

        #Split the data for 50 and 5 multi 
        python split_the_dataset_bm_multimodel.py $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/tf_peaks_$TF_NAME\_50_act.pkl $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/tf_peaks_$TF_NAME\_5_act.pkl $i $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/tf_peaks_$TF_NAME\_fasta.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME\_50_TFs.h5 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME\_5_TFs.h5 True
        echo "Done splitting the data for Binding partners into train/validation/test"
        
        #Train the multi-model with 50 outputs (binding partners of TF) (checked)
        python Run_String_Analysis_Training.py $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME\_50_TFs.h5 $MODEL_WEIGHTS_FOLDER/$TF_NAME\_50_multimodel_weights_$i 15 99 0.003 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels_50.pkl
        echo "Done training the multimodel with 50 outputs"
        
        #Train the multi-model with 5 outputs (binding partners of TF) (checked)
        python Run_String_Analysis_Training.py $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME\_5_TFs.h5 $MODEL_WEIGHTS_FOLDER/$TF_NAME\_5_multimodel_weights_$i 15 99 0.003 $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels_5.pkl
        echo "Done training the multimodel with 5 outputs"
        
        #Get a new data matrix for indiv model - without peaks that were used to train multi model (checked)
        python remove_training_data.py ../data/matrices/matrix2d.ReMap+UniBind.sparse.npz  $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/h5_files/tf_peaks_$TF_NAME\_50_TFs.h5 ../data/idx_files/regions_idx.pickle.gz ../data/idx_files/tfs_idx.pickle.gz ../data/final_df_dropped_string_real.pkl
        echo "Got a new data matrix for an indiv model"

        mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i
        
        #Get fasta and act files for the individual TF model (checked)
        python get_data_for_TF_subsample_positives_old.py $TF_NAME ../data/final_df_dropped_string_real.pkl ../data/sequences/sequences.200bp.fa 1000 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl #note it was 1000 subsampling!
        echo "Got fasta and act files for the indiv model by subsampling"
        
        mkdir -p $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files

        #Split the data for the individual model into train/validation/test (checked)
        python split_the_dataset.py $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_act.pkl 1 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/$TF_NAME\_fasta.pkl 0.1 0.1 $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files/$TF_NAME\_tl.h5 True
        echo "Done splitting the data for the indiv model into train/validation/test"
        
        #Perform training of an individual TF model with TL weights from multi-model (with inter partners)
        #and testing (checked)
        python Run_Analysis_String_Transfer_Learning.py $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files $MODEL_WEIGHTS_FOLDER/$TF_NAME\_indiv_weights_50_$i $MODEL_WEIGHTS_FOLDER/$TF_NAME\_50_multimodel_weights_$i 10 100 0.0003 50 $RESULTS_FOLDER/$TF_NAME\_50_TFs_$i True False $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels_50.pkl
        echo "Trained the individual model with TL (50 TFs) and done testing"
        
        #Perform training of an individual TF model with TL weights from multi-model (with inter partners)
        #and testing (checked)
        python Run_Analysis_String_Transfer_Learning.py $TRAIN_DATA_FOLDER/$TF_NAME\_indiv_$i/h5_files $MODEL_WEIGHTS_FOLDER/$TF_NAME\_indiv_weights_5_$i $MODEL_WEIGHTS_FOLDER/$TF_NAME\_5_multimodel_weights_$i 10 100 0.0003 5 $RESULTS_FOLDER/$TF_NAME\_5_TFs_$i True False $TRAIN_DATA_FOLDER/$TF_NAME\_multi_$i/$TF_NAME\_target_labels_5.pkl
        echo "Trained the individual model with TL (5 TFs) and done testing"

done