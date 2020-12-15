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
Input_h5_folder_with_test = sys.argv[2]
Output_weights_folder = sys.argv[3] #output folder with weights (weights_indiv_RELA/RELA_real)
TL_weights = sys.argv[4] #weights of multi model for tl (String_weights/RELA_real_model_weights/model_epoch_4_.pth)
Num_epochs = sys.argv[5] #number of epochs (10)
Batch_size = sys.argv[6] #batch size to use (100)
Learning_rate = sys.argv[7] #learning rate (0.0003)
Number_of_partners = sys.argv[8] #(10) - just the number of classes in the multimodel
Output_folder_for_metrics = sys.argv[9] #training_metrics_string_RELA
Do_TL = sys.argv[10]

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
    
mccoef = {}

tf_h5_files = os.listdir(Input_h5_folder_with_test)

for tf in tf_h5_files:
    tf_name = tf.split(".")[0]
    print("Analyzing %s" % tf_name)
    
    dataloaders, target_labels, train_out = load_datas(Input_h5_folder_with_test + "/" + tf, batch_size)
    target_labels = [i.decode("utf-8") for i in target_labels]
    num_classes = len(target_labels) #number of classes
    
    weights_file = os.listdir(Output_weights_folder + "/"+tf_name+"_weights/")[0]
    
    model = ConvNetDeep(num_classes, weight_path=Output_weights_folder + "/"+tf_name+"_weights/" + weights_file)
    #model = ConvNetDeep(num_classes, weight_path=weights)
    model.to(device);
    model.eval();
    
    labels_E_train, outputs_E_train = run_test(model, dataloaders['train'], device)
    labels_E_valid, outputs_E_valid = run_test(model, dataloaders['valid'], device)
    labels_E_test, outputs_E_test = run_test(model, dataloaders['test'], device)
    
    labels_E = np.concatenate((labels_E_train, labels_E_valid, labels_E_test))
    outputs_E = np.concatenate((outputs_E_train, outputs_E_valid, outputs_E_test))
    
    #get precision, recall and accuracy values
    ####################################################################################
    _, _, _, mcorcoef = compute_single_metrics(labels_E[:,0], outputs_E[:,0])
    mccoef[tf_name] = mcorcoef
    
    print("TF %s has cor coef: %10.2f" % (tf_name, mcorcoef))
    
if not os.path.exists(Output_folder_for_metrics):
            os.makedirs(Output_folder_for_metrics)
        
with open(Output_folder_for_metrics + '/mccoef.pkl', 'wb') as f:
    pickle.dump(mccoef, f)
    
print("Done!")