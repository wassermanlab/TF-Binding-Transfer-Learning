import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import collections
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from collections import OrderedDict
import os
import pickle
from models import *

import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch import relu, sigmoid
import torch.nn.modules.activation as activation
import matplotlib

matplotlib.use('Agg')
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn import metrics
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.ticker as ticker
import copy

import h5py
import kipoi
#import seaborn as sns

############################################################
#function for loading the dataset
############################################################
def load_datas(path_h5, batch_size):
    data = h5py.File(path_h5, 'r')
    dataset = {}
    dataloaders = {}
    #Train data
    dataset['train'] = torch.utils.data.TensorDataset(torch.Tensor(data['train_in']), 
                                                     torch.Tensor(data['train_out']))
    dataloaders['train'] = torch.utils.data.DataLoader(dataset['train'], 
                                                      batch_size=batch_size, shuffle=True,
                                                      num_workers=4)
    
    #Validation data
    dataset['valid'] = torch.utils.data.TensorDataset(torch.Tensor(data['valid_in']), 
                                                     torch.Tensor(data['valid_out']))
    dataloaders['valid'] = torch.utils.data.DataLoader(dataset['valid'], 
                                                      batch_size=batch_size, shuffle=True,
                                                      num_workers=4)
    
    #Test data
    dataset['test'] = torch.utils.data.TensorDataset(torch.Tensor(data['test_in']), 
                                                     torch.Tensor(data['test_out']))
    dataloaders['test'] = torch.utils.data.DataLoader(dataset['test'], 
                                                      batch_size=batch_size, shuffle=True,
                                                     num_workers=4)
    print('Dataset Loaded')
    target_labels = list(data['target_labels'])
    train_out = data['train_out']
    return dataloaders, target_labels, train_out

############################################################
#function to convert sequences to one hot encoding
#taken from Basset github repo
############################################################
def dna_one_hot(seq, seq_len=None, flatten=True):
    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq)-seq_len) // 2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq)) // 2

    seq = seq.upper()

    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    #  dtype='int8' fails for N's
    seq_code = np.zeros((4,seq_len), dtype='float16')
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:,i] = 0.25
        else:
            try:
                seq_code[int(seq[i-seq_start]),i] = 1
            except:
                seq_code[:,i] = 0.25

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_code = seq_code.flatten()[None,:]

    return seq_code

############################################################
#function to compute positive weights for BCEWithLogits
#pos weigths for class - num of neg examples/num of pos examples
############################################################
def compute_positive_weights(train_dataset, device):
    counts = np.array(train_dataset)
    pos_samples = counts.sum(axis=0)
    all_samples = np.full(counts.shape[1], counts.shape[0])
    neg_samples = all_samples - pos_samples
    pos_weights = neg_samples/pos_samples
    pos_weights = torch.from_numpy(pos_weights)
    pos_weights = pos_weights.float().to(device)
    return pos_weights

############################################################
#get AUPCR for predictions
############################################################
def get_aucpr(scores, labels):

    import numpy

    # Initialize #
    TPA = 0
    TPB = 0
    FPA = 0
    FPB = 0
    points = []
    TP_dict = {}    
    #paired_list = zip(scores, labels)
    #paired_list.sort(key=lambda x: x[0], reverse=True)
    paired_list = sorted(list(zip(scores, labels)), key=lambda x: x[0], reverse=True)
    total_positives = sum(labels)

    for cutoff, label in paired_list:
        TP_dict.setdefault(cutoff, [0,0])
        if label:
            TP_dict[cutoff][0] += 1
        else:
            TP_dict[cutoff][1] += 1

    sorted_cutoffs = sorted(list(TP_dict.keys()), reverse=True)

    TPB = TP_dict[sorted_cutoffs[0]][0]
    FPB = TP_dict[sorted_cutoffs[0]][1]

    # Initialize #
    points.extend(interpolate(0, TPB, 0, FPB, total_positives))

    for cutoff in range(1, len(sorted_cutoffs)):
        TPA += TP_dict[sorted_cutoffs[cutoff - 1]][0]
        TPB = TPA + TP_dict[sorted_cutoffs[cutoff]][0]
        FPA += TP_dict[sorted_cutoffs[cutoff - 1]][1]
        FPB = FPA + TP_dict[sorted_cutoffs[cutoff]][1]
        p = interpolate(TPA, TPB, FPA, FPB, total_positives)
        points.extend(p)

    x, y = list(zip(*points))

    return numpy.trapz(x=x, y=y)

############################################################
def interpolate(TPA, TPB, FPA, FPB, total_positives):

    # Initialize #
    points = []
    TPA = float(TPA)
    TPB = float(TPB)
    FPA = float(FPA)
    FPB = float(FPB)

    if (TPA - TPB) != 0:
        skew = (FPB-FPA)/(TPB-TPA)
        for x in range(int(TPB) - int(TPA) + 1):
            if (TPA + x + FPA + skew * x) > 0:
                points.append(((TPA + x) / total_positives, (TPA + x) / (TPA + x + FPA + skew * x)))

    return points

############################################################
#functions to compute the metrics
############################################################
def compute_single_metrics(labels, outputs):
    TP = np.sum(((labels == 1) * (np.round(outputs) == 1)))
    FP = np.sum(((labels == 0) * (np.round(outputs) == 1)))
    TN = np.sum(((labels == 0) * (np.round(outputs) == 0)))
    FN = np.sum(((labels == 1) * (np.round(outputs) == 0)))
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    mcorcoef = matthews_corrcoef(labels, np.round(outputs))
    
    return precision, recall, accuracy, mcorcoef

############################################################
#functions to compute the metrics
############################################################
def compute_metrics(labels, outputs, save=None):
    TP = np.sum(((labels == 1) * (np.round(outputs) == 1)))
    FP = np.sum(((labels == 0) * (np.round(outputs) == 1)))
    TN = np.sum(((labels == 0) * (np.round(outputs) == 0)))
    FN = np.sum(((labels == 1) * (np.round(outputs) == 0)))
    print('TP : {} FP : {} TN : {} FN : {}'.format(TP, FP, TN, FN))
    #plt.bar(['TP', 'FP', 'TN', 'FN'], [TP, FP, TN, FN])
    
    #if save:
    #    plt.savefig(save)
    #else:
    #    plt.show()
    
    try:
        print('Roc AUC Score : {:.2f}'.format(roc_auc_score(labels, outputs)))
        print('AUPRC {:.2f}'.format(average_precision_score(labels, outputs)))
    except ValueError:
        pass
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print('Precision : {:.2f} Recall : {:.2f} Accuracy : {:.2f}'.format(precision, recall, accuracy))

############################################################    
#function to test the performance of the model
############################################################
def run_test(model, dataloader_test, device):
    running_outputs = []
    running_labels = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for seq, lbl in dataloader_test:
            #seq = seq.permute(0, 1, 3, 2).to(device)
            #seq = seq.squeeze(-1)
            seq = seq.to(device)
            out = model(seq)
            out = sigmoid(out.detach().cpu()) #for BCEWithLogits
            running_outputs.extend(out.numpy()) #for BCEWithLogits
            #running_outputs.extend(out.detach().cpu().numpy())
            running_labels.extend(lbl.numpy())
    return np.array(running_labels), np.array(running_outputs)

############################################################    
#function to get activations for the sequences
############################################################
def get_motifs(data_loader, model, device):
    running_outputs = []
    running_activations = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for seq, lbl in tqdm(data_loader, total=len(data_loader)):
            #seq = seq.permute(0, 1, 3, 2).to(device)
            #seq = seq.squeeze(-1)
            seq = seq.to(device)
            out, act = model(seq)
            out = sigmoid(out.detach().cpu()) #for BCEWithLogits
            running_outputs.extend(out.numpy()) #for BCEWithLogits
            running_activations.extend(act.cpu().numpy())
            
    return np.array(running_outputs), np.array(running_activations)
    #return running_outputs, running_activations

############################################################
#function to plot bar plot of results
############################################################
def plot_results(labels, outputs, targets):

    TP = np.sum(((labels == 1) * (np.round(outputs) == 1)),axis=0)
    FP = np.sum(((labels == 0) * (np.round(outputs) == 1)),axis=0)
    TN = np.sum(((labels == 0) * (np.round(outputs) == 0)),axis=0)
    FN = np.sum(((labels == 1) * (np.round(outputs) == 0)),axis=0)
    
    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
        title='Transcription factors'),
        yaxis=dict(
        title='Sequences'),
        font=dict(
            size = 18,
            color='#000000'
        ))

    fig = go.Figure(data=[
            go.Bar(name='TP', x=targets, y=TP),
            go.Bar(name='FP', x=targets, y=FP),
            go.Bar(name='TN', x=targets, y=TN),
            go.Bar(name='FN', x=targets, y=FN)
        ], layout=layout)
        # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.show()

############################################################
#function to plot bar plot of true label ratios
############################################################
def plot_ratios(labels, targets):
    Counts = labels.sum(0)
    Zeros = labels.shape[0] - Counts

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
        title='Transcription factors'),
        yaxis=dict(
        title='Sequences'),
        font=dict(
            size = 18,
            color='#000000'
        ))

    fig = go.Figure(data=[
            go.Bar(name='Ones', x=targets, y=Counts),
            go.Bar(name='Zeros', x=targets, y=Zeros)
        ], layout=layout)
        # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.show()

############################################################    
#function to show training curve
#save - place to save the figure
############################################################
def showPlot(points, points2, title, ylabel, save=None):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.plot(points2)
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.title(title)
    
    if save:
        plt.savefig(save)
    else:
        plt.show()
        
############################################################    
#name is self-explanatory
############################################################
def convert_onehot_back_to_seq(dataloader):
    sequences = []
    code = ['A', 'C', 'G', 'T']
    for seqs, labels in tqdm(dataloader, total=len(dataloader)):
        x = seqs.permute(0, 1, 3, 2)
        x = x.squeeze(-1)
        for i in range(x.shape[0]):
            seq = ""
            for j in range(x.shape[-1]):
                try:
                    seq = seq + code[int(np.where(x[i,:,j] == 1)[0])]
                except:
                    print("error")
                    print(x[i,:,j])
                    print(np.where(x[i,:,j] == 1))
                    break
            sequences.append(seq)
    return sequences

############################################################
#function to train a model
############################################################
def train_model(train_loader, test_loader, model, device, criterion, optimizer, num_epochs,
               weights_folder, name_ind, verbose):
    
    total_step = len(train_loader)
    
    train_error = []
    test_error = []
    
    train_fscore = []
    test_fscore = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 1

    for epoch in range(num_epochs):
        
        model.train() #tell model explicitly that we train
        
        logs = {}
        
        running_loss = 0.0
        running_fbeta = 0.0
        
        for seqs, labels in train_loader:
            #x = seqs.permute(0, 1, 3, 2).to(device)
            #x = x.squeeze(-1)
            x = seqs.to(device)
            labels = labels.to(device)
            
            #zero the existing gradients so they don't add up
            optimizer.zero_grad()

            # Forward pass
            #outputs, act, idx = model(x)
            outputs = model(x)
            loss = criterion(outputs, labels) 
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
            #f-beta score
            #labels = labels.cpu()
            #outputs = outputs.cpu()
            #fbeta_score = metrics.fbeta_score(labels, outputs > 0.5, 2, average='samples')
            #running_fbeta += fbeta_score
            
        #scheduler.step() #learning rate schedule
        
        #save training loss to file
        epoch_loss = running_loss / len(train_loader)
        logs['train_log_loss'] = epoch_loss
        train_error.append(epoch_loss)
        
        #epoch_fscore = running_fbeta / len(train_loader)
        #train_fscore.append(epoch_fscore)

        #calculate test (validation) loss for epoch
        test_loss = 0.0
        test_fbeta = 0.0
        
        with torch.no_grad(): #we don't train and don't save gradients here
            model.eval() #we set forward module to change dropout and batch normalization techniques
            for seqs, labels in test_loader:
                #x = seqs.permute(0, 1, 3, 2).to(device)
                #x = x.squeeze(-1)
                x = seqs.to(device)
                y = labels.to(device)
                #outputs, act, idx = model(x)
                model.eval() #we set forward module to change dropout and batch normalization techniques
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                
                #labels = labels.cpu()
                #outputs = outputs.cpu()
                #fbeta_score = metrics.fbeta_score(labels, outputs > 0.5, 2, average='samples')
                #test_fbeta += fbeta_score

        test_loss = test_loss / len(test_loader) #len(test_loader.dataset)
        logs['test_log_loss'] = test_loss
        test_error.append(test_loss)
        
        #test_fbeta = test_fbeta/len(test_loader)
        #test_fscore.append(test_fbeta)
        
        if verbose:
            print ('Epoch [{}], Current Train Loss: {:.5f}, Current Val Loss: {:.5f}' 
                       .format(epoch+1, epoch_loss, test_loss))
            
        if test_loss < best_loss_valid:
            best_loss_valid = test_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            #torch.save(best_model_wts, weights_folder + "/"+"model_epoch_"+str(epoch+1)+"_"+
            #           name_ind+".pth") #weights_folder, name_ind
            #print ('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}'.format(epoch+1, best_loss_valid))
        
        #if (epoch+1)%5 == 0:
            #model_wts = copy.deepcopy(model.state_dict())
            #torch.save(model_wts, weights_folder + "/"+"model_epoch_"+str(epoch+1)+"_"+
            #           name_ind+".pth") #weights_folder, name_ind

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, weights_folder + "/"+"model_epoch_"+str(best_epoch+1)+"_"+
                       name_ind+".pth") #weights_folder, name_ind
    
    #return model, best_loss_valid
    return model, train_error, test_error, train_fscore, test_fscore


############################################################
#function for training individual models
#for each TF:
#tf_h5_files_folder - folder with h5 files with train/valid/test
#TL - use transfer learning or not, default False
#(if specified as True you also need to provide weights of the model)
#Doesn't return models but saves their weights
#output_folder - Saves the weights of individual models in the provided folder output_folder
#image_folder - Saves train images in the folder Images_indiv
############################################################
def train_individual_TF_models(tf_h5_files_folder, device, output_folder, image_folder,
                               num_epochs, learning_rate, batch_size, TL=False, num_class_orig=None,
                               weights=None, verbose=False):
    
    tf_h5_files = os.listdir(tf_h5_files_folder)
    
    for tf in tf_h5_files:
        tf_name = tf.split(".")[0]
        print("Analyzing %s" % tf_name)
    
        #loading the data
        dataloaders, target_labels, train_out = load_datas(tf_h5_files_folder +"/" + tf, batch_size)
    
        #skip TFs with less than 500 sequences in the train data
        #subject to change
        #if len(dataloaders["train"].dataset) < 500:
        #    print("Not enough train data for TF %s" % tf_name)
        #    continue
    
        print("TF %s has %d training examples" % (tf_name, len(dataloaders["train"].dataset)))
    
        #decode label names
        target_labels = [i.decode("utf-8") for i in target_labels]
    
        num_classes = len(target_labels) #number of classes

        if TL:
            assert weights, "No weights specified."
            #because we load weights from the model that was trained on 50 classes
            model = ConvNetDeep(num_class_orig, weight_path=weights)
            #model = ConvNetDeepBNAfterRelu(num_class_orig, weight_path=weights)
            #model = DanQ(num_class_orig, weight_path=weights)
            
            #reinitialize the last layer of the model (OUR WAY)
            model.d6 = nn.Linear(1000, num_classes)
            
            #for DanQ
            #model.Linear2 = nn.Linear(925, num_classes)
            
            #freezing way (comment after!)
            #for child in list(model.children())[:12]:
            #    for param in child.parameters():
            #        param.requires_grad = False
                    
            model = model.to(device)
        else:
            model = ConvNetDeep(num_classes)
            #model = ConvNetDeepBNAfterRelu(num_classes)
            #model = DanQ(num_classes)
            model = model.to(device)
    
        #loss function is Binary cross entropy with logits
        criterion = nn.BCEWithLogitsLoss() #- no weights
        #an optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        if not os.path.exists(output_folder + "/" + tf_name + "_weights"):
            os.makedirs(output_folder + "/" + tf_name + "_weights")
    
        #train the model
        model, train_error, test_error, train_fscore, test_fscore = train_model(dataloaders['train'], 
                                                                                dataloaders['valid'], 
                                                                                model, device, criterion,  optimizer, 
                                                                                num_epochs, 
                                                                                output_folder + "/" + tf_name + "_weights", 
                                                                                "", verbose=False) 
    
        #if not os.path.exists(image_folder + "/" + tf_name):
        #    os.makedirs(image_folder + "/" + tf_name)
        
        #save training plot
        #showPlot(train_error, test_error, "Loss trend", "Loss", save=image_folder + "/" + "image_" + tf_name + ".png")
        
        if verbose:
            print("Done with %s" % tf_name)
            
############################################################
#function for evaluating individual models
############################################################
def test_individual_TF_models(h5_folder, device, weights, batch_size, use_orig_model=False, num_class_orig=None,
                              target_labels_file=None,
                              weights_old=None, verbose=True):
    
    if use_orig_model:
        pkl_file = open(target_labels_file, 'rb')
        target_labels_old = pickle.load(pkl_file)
        pkl_file.close()
        
        aucroc_old = {}
        aucprc_old = {}
        prec_old = {}
        rec_old = {}
        accur_old = {}
        mccoef_old = {}

    aucroc = {}
    aucprc = {}
    prec = {}
    rec = {}
    accur = {}
    mccoef = {}

    tf_h5_files = os.listdir(h5_folder)

    for tf in tf_h5_files:
        tf_name = tf.split(".")[0]
        print("Analyzing %s" % tf_name)
        
        if use_orig_model:
            ind = np.where(np.array(target_labels_old) == tf_name)[0]
    
        #loading the data
        dataloaders, target_labels, train_out = load_datas(h5_folder + "/" + tf, batch_size)
    
        #skip TFs with less than 1000 sequences in the train data
        #subject to change
        #if len(dataloaders["train"].dataset) < 500:
        #    print("Not enough train data for TF %s" % tf_name)
        #    continue
    
        #decode label names
        target_labels = [i.decode("utf-8") for i in target_labels]
    
        num_classes = len(target_labels) #number of classes

        #because we load weights from the model that was trained on 50 classes
        if use_orig_model:
            model_old = ConvNetDeep(num_class_orig, weight_path=weights_old)
            #model_old = ConvNetDeepBNAfterRelu(num_class_orig, weight_path=weights_old)
            #model_old = DanQ(num_class_orig, weight_path=weights_old)
            model_old.to(device);
            model_old.eval();
            
        weights_file = os.listdir(weights + "/"+tf_name+"_weights/")[0]
        
        model = ConvNetDeep(num_classes, weight_path=weights + "/"+tf_name+"_weights/" + weights_file)
        #model = ConvNetDeepBNAfterRelu(num_classes, weight_path=weights + "/"+tf_name+"_weights/" + weights_file)
        #model = DanQ(num_classes, weight_path=weights + "/"+tf_name+"_weights/" + weights_file)
        
        model.to(device);
        model.eval();
        
        #get predictions
        labels_E, outputs_E = run_test(model, dataloaders['test'], device)
        if use_orig_model and len(ind) == 1:
            labels_E_old, outputs_E_old = run_test(model_old, dataloaders['test'], device)
    
        #get auc_values
        ####################################################################################
        nn_fpr, nn_tpr, threshold = metrics.roc_curve(labels_E[:,0], outputs_E[:,0])
        roc_auc_nn = metrics.auc(nn_fpr, nn_tpr)
        aucroc[tf_name] = roc_auc_nn 
        if use_orig_model and len(ind) == 1:
            nn_fpr_old, nn_tpr_old, threshold = metrics.roc_curve(labels_E_old[:,0], outputs_E_old[:,ind[0]])
            roc_auc_nn_old = metrics.auc(nn_fpr_old, nn_tpr_old)
            aucroc_old[tf_name] = roc_auc_nn_old
    
        #get auprc values
        ####################################################################################
        precision_nn, recall_nn, thresholds = metrics.precision_recall_curve(labels_E[:,0], outputs_E[:,0])
        pr_auc_nn = metrics.auc(recall_nn, precision_nn)
        aucprc[tf_name] = pr_auc_nn
        if use_orig_model and len(ind) == 1:
            precision_nn_old, recall_nn_old, thresholds = metrics.precision_recall_curve(labels_E_old[:,0], 
                                                                                     outputs_E_old[:,ind[0]])
            pr_auc_nn_old = metrics.auc(recall_nn_old, precision_nn_old)
            aucprc_old[tf_name] = pr_auc_nn_old
        
        #get precision, recall and accuracy values
        ####################################################################################
        precision, recall, accuracy, mcorcoef = compute_single_metrics(labels_E[:,0], outputs_E[:,0])
        prec[tf_name] = precision
        rec[tf_name] = recall
        accur[tf_name] = accuracy
        mccoef[tf_name] = mcorcoef
    
        if use_orig_model and len(ind) == 1:
            precision_old, recall_old, accuracy_old, mcorcoef_old = compute_single_metrics(labels_E_old[:,0], 
                                                                         outputs_E_old[:,ind[0]])
            prec_old[tf_name] = precision_old
            rec_old[tf_name] = recall_old
            accur_old[tf_name] = accuracy_old
            mccoef_old[tf_name] = mcorcoef_old
        if verbose:
            print("TF %s has precision: %10.2f" % (tf_name, precision))
            print("TF %s has recall: %10.2f" % (tf_name, recall))
            print("TF %s has accuracy: %10.2f" % (tf_name, accuracy))
            print("TF %s has cor coef: %10.2f" % (tf_name, mcorcoef))
            if use_orig_model and len(ind) == 1:
                print("For multi-model TF %s has precision: %10.2f" % (tf_name, precision_old))
                print("For multi-model TF %s has recall: %10.2f" % (tf_name, recall_old))
                print("For multi-model TF %s has accuracy: %10.2f" % (tf_name, accuracy_old))
                print("For multi-model TF %s has cor coef: %10.2f" % (tf_name, mcorcoef_old))
    
        print("Done with %s" % tf_name)
        
    if use_orig_model:
        return (aucroc, aucprc, prec, rec, accur, mccoef, aucroc_old, aucprc_old, prec_old, rec_old, accur_old, mccoef_old)
    else:
        return (aucroc, aucprc, prec, rec, accur, mccoef)

#####################################################################################
"""
Code adapted from : https://github.com/smaslova/AI-TAC/blob/3d92cecb6e6b75d0ba7f09054a3a487307f62055/code/plot_utils.py#L391
"""

def get_memes(activations, sequences, y, output_file_path):
    """
    Extract pwm for each filter and save it as a .meme file ( PWM ) using the activations and the original sequences.
    params :
        actvations (np.array) : (N*N_filters*L) array containing the ourput for each filter and selected sequence of the test set
        sequnces (np.array) : (N*4*200) selected sequences (ACGT)
        y (np.array) : (N*T) original target of the selected sequnces
        output_file_path (str) : path to directory to store the resulting pwm meme file
    """
    #find the threshold value for activation
    activation_threshold = 0.5*np.amax(activations, axis=(0,2))

    # Get the number of filters
    N_FILTERS = activations.shape[1]

    #pad sequences:
    #npad = ((0, 0), (0, 0), (9, 9))
    #sequences = np.pad(sequences, pad_width=npad, mode='constant', constant_values=0)

    pwm = np.zeros((N_FILTERS, 4, 19))
    pfm = np.zeros((N_FILTERS, 4, 19))
    nsamples = activations.shape[0]

    OCR_matrix = np.zeros((N_FILTERS, y.shape[0]))
    activation_indices = []
    activated_OCRs = np.zeros((N_FILTERS, y.shape[1]))
    n_activated_OCRs = np.zeros(N_FILTERS)
    total_seq = np.zeros(N_FILTERS)

    for i in tqdm(range(N_FILTERS)):
        #create list to store 19 bp sequences that activated filter
        act_seqs_list = []
        act_OCRs_tmp = []
        for j in range(nsamples):
            # find all indices where filter is activated
            indices = np.where(activations[j,i,:] > activation_threshold[i])

            #save ground truth peak heights of OCRs activated by each filter
            if indices[0].shape[0]>0:
                act_OCRs_tmp.append(y[j, :])
                OCR_matrix[i, j] = 1

            for start in indices[0]:
                activation_indices.append(start)
                end = start+19
                act_seqs_list.append(sequences[j,:,start:end])

        #convert act_seqs from list to array
        if act_seqs_list:
            act_seqs = np.stack(act_seqs_list)
            pwm_tmp = np.sum(act_seqs, axis=0)
            pfm_tmp=pwm_tmp
            total = np.sum(pwm_tmp, axis=0)
            pwm_tmp = np.nan_to_num(pwm_tmp/total)

            pwm[i] = pwm_tmp
            pfm[i] = pfm_tmp

            #store total number of sequences that activated that filter
            total_seq[i] = len(act_seqs_list)

            #save mean OCR activation
            act_OCRs_tmp = np.stack(act_OCRs_tmp)
            activated_OCRs[i, :] = np.mean(act_OCRs_tmp, axis=0)

            #save the number of activated OCRs
            n_activated_OCRs[i] = act_OCRs_tmp.shape[0]


    activated_OCRs = np.stack(activated_OCRs)

    #write motifs to meme format
    #PWM file:
    meme_file = open(output_file_path, 'w')
    meme_file.write("MEME version 4 \n")

    print('Saved PWM File as : {}'.format(output_file_path))

    for i in range(0, N_FILTERS):
        if np.sum(pwm[i,:,:]) >0:
            meme_file.write("\n")
            meme_file.write("MOTIF filter%s \n" % i)
            meme_file.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i,:,:], axis=0)))


        for j in range(0, 19):
              if np.sum(pwm[i,:,j]) > 0:
                meme_file.write(str(pwm[i,0,j]) + "\t" + str(pwm[i,1,j]) + "\t" + str(pwm[i,2,j]) + "\t" + str(pwm[i,3,j]) + "\n")

    meme_file.close()
    
##########################################
#from:
#https://github.com/Etienne-Meunier/Basset_E/blob/master/Interpretation/Single_Cell_Interpretation/Filter%20Importance.py
##########################################
def get_average_act(self, input, output) :
    """Pytorch Hook that will get the average activation for each layer on the current batch
        which can be usefull to get the average usage of the filter
    Args :
        self (pytorch layer): the layer the hook is attached to
        input (tensor): current input tensor of the layer
        output (tensor): current output tensor of the layer
    """
    if self.mode[0] == 'Baseline' :
        self.register_buffer('average_activation', output.mean(0).mean(1))
        
##########################################
#from:
#https://github.com/Etienne-Meunier/Basset_E/blob/master/Interpretation/Single_Cell_Interpretation/Filter%20Importance.py
##########################################
def nullify_filter_strict(self, input, output) :
    """Pytorch Hook that will nullify the output of one of the filter indicated in mode for that layer
    Args :
        self (pytorch layer): the layer the hook is attached to
        input (tensor): current input tensor of the layer
        output (tensor): current output tensor of the layer
    """
    if self.mode[0] == 'Compare' :
        output[:,self.mode[1],:] = 0
    
##########################################
#function to compute filter importance
#from:
#https://github.com/Etienne-Meunier/Basset_E/blob/master/Interpretation/Single_Cell_Interpretation/Filter%20Importance.py
##########################################
def compute_filter_importance(model, dataloader, target_labels, out_shape, output_dir) :
    """Main sequences that go through the val set, nullifying layer one by one to get the average impact of each layer
    Args:
        B (Pytorch Model): Model to analyse
        dataloader (dataloader): dataloader with selected data.
        target_lanels (list:string): columns names in the target
        output_dir (str): directory where to store the average impact and average activation csv files
    """
    with torch.no_grad() : # Remove grad computation for speed
        model.cuda()
        model.rl1.register_forward_hook(get_average_act)
        model.rl1.register_forward_hook(nullify_filter_strict)

        N_FILTERS = model.c1.weight.shape[0]
        # We accumulate the impacts and activations in those tensors
        average_activations = torch.zeros((N_FILTERS)) # One average activation by filter
        average_impacts = torch.zeros((N_FILTERS, out_shape)) # One impact by filter and by TF

        loader = tqdm(dataloader, total=len(dataloader))
        sigmoid = nn.Sigmoid()
        for X, y in loader : # For all the selected sequences
            X = X.cuda()
            model.rl1.mode = ('Baseline',)
            baseline  = sigmoid(model(X)) # Compute the average activation
            temp_imp = []
            for i in range(N_FILTERS) : # nullify filters one by one
                model.rl1.mode = ('Compare', i)
                temp_imp.append(torch.mean((sigmoid(model(X))-baseline)**2, axis=0).detach().cpu()) # compute difference with baseline
            average_impacts += torch.stack(temp_imp) # Add to the previous batch differences
            average_activations += model.rl1.average_activation.detach().cpu()
        average_activations = average_activations/len(dataloader)
        average_impacts = average_impacts/len(dataloader)

        # Create dataframe and export to csv
        index=['filter{}'.format(idx) for idx in range(100)]
        average_activations_df = pd.DataFrame(average_activations.numpy())
        average_activations_df.index = index

        average_impacts_df = pd.DataFrame(average_impacts.numpy(),
                    columns=target_labels,
                    index=['filter{}'.format(idx) for idx in range(100)])
        average_impacts_df.index = index

        average_activations_df.to_csv(output_dir+'average_activations.csv')
        average_impacts_df.to_csv(output_dir+'average_impacts.csv')