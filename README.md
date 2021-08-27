# README

## Description of notebooks and scripts

### Ipython notebooks

1. **Explore_matrices_and_get_data_for_multi_model.ipynb** - in this notebook we explore the UniBind+Remap matrix and collect data to train a 50 TF multi-model;
2. **Train_multi_model.ipynb** - in this notebook we train a CNN multi-model with 50 TFs and generate the results;
3. **Train_multi_model_DanQ.ipynb** - in this notebook we train a DanQ multi-model with 50 TFs and generate the results;
4. **TL_exploring_pentad_TFs.ipynb** - in this notebook we explore the results of training with biologically relevant groups (with TF pentad) using a CNN model;
5. **TL_exploring_pentad_TFs_DanQ.ipynb** - in this notebook we explore the results of training with biologically relevant groups (with TF pentad) using a DanQ model;
6. **TL_effect_of_multimodels.ipynb** - effect of different multi-models (5 vs 50) on the individual model performance (for pentad TFs);
7. **TL_individual_models.ipynb** - plotting the results for the individual models vs multi-model; plotting the box plots for the effect of TL;
8. **TL_with_freezed_layers.ipynb** - inspecting the TL performance, when convolutional layers are freezed;
9. **Exploring_effect_of_BM_on_15_TFs.ipynb** - in this notebook we lot the effect of BM on TL for 15 TFs (3 TF for each of the pentad family);
10. **Model_interpretation.ipynb** - here we perform multi- and individual models interpretation by converting first layer convolutional filters into PWMs;
11. **Interpretation_of_models_finetuned_with_cofactors.ipynb** - interpreting individual models that were preinitialized with weights from multi-models trained on cofactors; 
12. **Data_size_effect_on_TL.ipynb** - exploring how TL affects the performance for different sub-sampled data sets;
13. **UMAP_Binding_heatmap_and_selecting_groups.ipynb** -  in this notebook we analyze the binding matrix by plotting UMAP plot and the heatmap of binding pattern similarities. Moreover, we select biologically relevant groups for TL;

### Bash scripts

1. **run_training_of_individual_models.sh** - run to train 148 individual TF models from scratch or using 50 TF multi-model to initialize weights;
2. **run_training_of_individual_models_FREEZING_LAYERS.sh** -  run to train 148 individual TF models from scratch or using 50 TF multi-model to initialize weights; this time convolutional layers are freezed;
3. **run_single_tf_refined_subsample_experiment.sh** - run to test TL boundaries by sub-sampling different numbers of positive regions;
4.  **run_BM_real_TFs_last_exp.sh** (runs **run_BM_tl_last_exp_corrected_remove.sh**) - trains models with TL for 15 TFs;
5. **run_BM_multimodel_TFs.sh** (runs **run_BM_multimodel.sh**) - perform TL using either 50 or 5 TF multi-model;
6. **run_BM_real_TFs.sh** (runs **run_BM_tl_subsample.sh**) - train individual models using TL with TFs from the same BM; for speed, subsample data sets to 1000 positives/negatives; also runs **run_BM_tl_subsample_DanQ.sh** - same as above but for DanQ;
7. **run_cofactors_real_TFs.sh** (runs **run_cofactor_tl_subsample.sh**) - train individual models using TL with TFs that are cofactors/STRING partners/low correlated TFs with the same BM; for speed, subsample data sets to 1000 positives/negatives;
8. **get_indiv_data_for_each_TF.sh** - get data splits for individual TFs;

### Python scripts

1. **split_the_dataset.py** - script that takes as input fasta files and labels and splits the data into train/validation/test sets; one-hot encodes (and reverse complements if required) the sequences; 
2. **Run_Analysis_Training.py** - trains a multi-model;
3. **Run_String_Analysis_Training.py** - same as above, but saves class labels in a separate file;
4. **remove_training_data.py** - script for removing data used to train a multi-model from the original TF binding matrix;
5. **Run_Analysis_Transfer_Learning.py** - trains individual TF models with/without TL and with/without testing; 
6. **Run_Analysis_String_Transfer_Learning.py** - same as above, but accepts class labels to use during the testing;
7. **Run_Analysis_Transfer_Learning_Subsampling.py** - same as above, but takes as input a specified test data set (used during testing TL boundaries); 
8. **get_data_for_TF.py** - script to build fasta and labels files for a specific TF;
9. **get_data_for_TF_subsample_positives_old.py** -  same as above, but subsamples data to a certain number of positives/negatives
10. **get_data_for_TF_subsample_positives.py** - same as above, but also subsamples a certain number of test sequences;
11. **Run_BM_Analysis.py** - generates fasta and labels files for a specific TF and binding mode; the final subsampled data cannot be less than 70,000 sequences;
12. **Run_BM_Analysis_LE.py** - same as above, but no restriction on subsampled data size; 
13. **Run_BM_Multimodel_Analysis.py** - same as above, but randomly samples 40,000 regions, and saves labels for 50 and 5 classes multi-model;
14. **Run_Cofactor_Analysis.py** - generates fasta and labels files for a specific TF and its biological group (cofactors, string, low correlated BM); the final subsampled data cannot be less than 70,000 sequences;
15. **Run_Cofactor_Multimodel_Analysis.py** - same as above, but randomly samples 40,000 regions, and saves labels for 50 and 5 classes multi-model;
16. **split_the_dataset_bm_multimodel.py** - splits data set for different multimodels;
17. **models.py** - python script with model architectures;
18. **tools.py** - python script with functions used to analyze the data;
19. **deeplift_scores.py** - python script to compute DeepLIFT importance scores using the Captum library;



