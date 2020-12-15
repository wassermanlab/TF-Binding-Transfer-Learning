from tools import *
from models import *
#import plotly.graph_objects as go
#import plotly.figure_factory as ff
from Bio.SeqUtils import GC
from Bio import SeqIO
import sys
import os
import pickle

Input_h5_folder = sys.argv[1] #folder with h5 file for indiv TF (RELA_indiv_tl_h5)
Output_weights_folder = sys.argv[2] #output folder with weights (weights_indiv_RELA/RELA_real)
TL_weights = sys.argv[3] #weights of multi model for tl (String_weights/RELA_real_model_weights/model_epoch_4_.pth)
Num_epochs = sys.argv[4] #number of epochs (10)
Batch_size = sys.argv[5] #batch size to use (100)
Learning_rate = sys.argv[6] #learning rate (0.0003)
Number_of_partners = sys.argv[7] #(10) - just the number of classes in the multimodel
Output_folder_for_metrics = sys.argv[8] #training_metrics_string_RELA
Do_TL = sys.argv[9]
Use_orig_model = sys.argv[10] #for testing (True/False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = int(Num_epochs)
batch_size = int(Batch_size)
learning_rate = float(Learning_rate)

weights_file_tl = os.listdir(TL_weights)[0]

if Do_TL == "True":
    train_individual_TF_models(Input_h5_folder, device, Output_weights_folder, "Images_indiv",
                           num_epochs, learning_rate, batch_size, TL=True, num_class_orig=int(Number_of_partners),
                           weights=TL_weights+"/"+weights_file_tl, verbose=True)
    print("Done with training TL")
else:
    train_individual_TF_models(Input_h5_folder, device, Output_weights_folder, "Images_indiv_notf",
                           num_epochs, learning_rate, batch_size, TL=False, weights=None, verbose=True)
    print("Done with training from scratch")

#weights_file = os.listdir(Output_weights_folder)[0]

if Use_orig_model == 'True':
    
    res = test_individual_TF_models(Input_h5_folder, device, Output_weights_folder, 
                                    batch_size, use_orig_model=True, num_class_orig=int(Number_of_partners),
                                    target_labels_file='../data/multi_model_target_labels.pkl',
                                    weights_old=TL_weights+"/"+weights_file_tl, verbose=True)
    print("Done with testing")
    aucroc, aucprc, prec, rec, accur, mccoef, aucroc_old, aucprc_old, prec_old, rec_old, accur_old, mccoef_old = res
    
else:
    
    res = test_individual_TF_models(Input_h5_folder, device, Output_weights_folder, 
                                batch_size, use_orig_model=False, verbose=True)
    print("Done with testing")
    aucroc, aucprc, prec, rec, accur, mccoef = res
    
if not os.path.exists(Output_folder_for_metrics):
            os.makedirs(Output_folder_for_metrics)

with open(Output_folder_for_metrics + '/aucroc.pkl', 'wb') as f:
    pickle.dump(aucroc, f)
    
with open(Output_folder_for_metrics + '/aucprc.pkl', 'wb') as f:
    pickle.dump(aucprc, f)
    
with open(Output_folder_for_metrics + '/prec.pkl', 'wb') as f:
    pickle.dump(prec, f)
    
with open(Output_folder_for_metrics + '/rec.pkl', 'wb') as f:
    pickle.dump(rec, f)
    
with open(Output_folder_for_metrics + '/accur.pkl', 'wb') as f:
    pickle.dump(accur, f)
    
with open(Output_folder_for_metrics + '/mccoef.pkl', 'wb') as f:
    pickle.dump(mccoef, f)
    
if Use_orig_model == 'True':
    with open(Output_folder_for_metrics + '/aucroc_old.pkl', 'wb') as f:
        pickle.dump(aucroc_old, f)
    with open(Output_folder_for_metrics + '/aucprc_old.pkl', 'wb') as f:
        pickle.dump(aucprc_old, f)
    with open(Output_folder_for_metrics + '/prec_old.pkl', 'wb') as f:
        pickle.dump(prec_old, f)
    with open(Output_folder_for_metrics + '/rec_old.pkl', 'wb') as f:
        pickle.dump(rec_old, f)
    with open(Output_folder_for_metrics + '/accur_old.pkl', 'wb') as f:
        pickle.dump(accur_old, f)
    with open(Output_folder_for_metrics + '/mccoef_old.pkl', 'wb') as f:
        pickle.dump(mccoef_old, f)
    
print("Done!")
