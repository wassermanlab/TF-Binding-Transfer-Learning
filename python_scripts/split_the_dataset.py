import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle
import sys
import h5py
import tqdm
import numpy.random as npr
from Bio import Seq
from tools import *

Matrix_pkl_file = sys.argv[1] #example - 'new_data/tf_peaks_50_noNs.pkl'
Seed = sys.argv[2] #seed for permutations; example - 1
Fasta_pkl_file = sys.argv[3] #example - new_data/fasta_sequences_50.pkl
test_pct = sys.argv[4] #fraction of test data; example - 0.1 (it means 10%)
valid_pct = sys.argv[5] #fraction of validation data; example - 0.1 (it means 10%)
Output_h5 = sys.argv[6] #example - my_own_h5_50.h5
Add_reverse_complement = sys.argv[7] #Add reverse complement or not - True/False

with open(Matrix_pkl_file, 'rb') as f:
    tf_peaks = pickle.load(f)

with open(Fasta_pkl_file, 'rb') as f:
    fasta_sequences = pickle.load(f)

####################################
#shuffle the sequence
####################################
npr.seed(int(Seed)) #setting a seed

order = npr.permutation(tf_peaks.index)
tf_peaks = tf_peaks.reindex(order) #labels
fasta_sequences = fasta_sequences[order] #sequences
print("Done shuffling")
####################################

test_count = int(0.5 + float(test_pct) * tf_peaks.shape[0])
valid_count = int(0.5 + float(valid_pct) * tf_peaks.shape[0])
train_count = tf_peaks.shape[0] - test_count - valid_count

print('%d training sequences ' % train_count)
print('%d test sequences ' % test_count)
print('%d validation sequences ' % valid_count)

####################################
#shuffle the sequence
####################################
i = 0
train_peaks, train_sequences = tf_peaks.iloc[i:i+train_count,:], fasta_sequences[i:i+train_count]
i += train_count
valid_peaks, valid_sequences = tf_peaks.iloc[i:i+valid_count,:], fasta_sequences[i:i+valid_count]
i += valid_count
test_peaks, test_sequences = tf_peaks.iloc[i:i+test_count,:], fasta_sequences[i:i+test_count]
####################################

####################################
#saving TF labels and sequences ids
#which we used for training/validation
####################################
target_labels = list(tf_peaks)
train_peaks.index = train_peaks.index.map(str)
valid_peaks.index = valid_peaks.index.map(str)
test_peaks.index = test_peaks.index.map(str)

train_headers = train_peaks.index
valid_headers = valid_peaks.index

target_labels = [a.encode('utf8') for a in target_labels]
train_headers = [a.encode('utf8') for a in train_headers]
valid_headers = [a.encode('utf8') for a in valid_headers]
####################################

if Add_reverse_complement == "True":
    ####################################
    #add reverse complement ids
    #to train and valid dfs
    ####################################
    #train
    #change index to make it reverse
    new_index = train_peaks.index.map(lambda x: x + "_r")
    #copy of the original data frame
    interim = train_peaks.copy()
    interim.index = new_index
    #new df with reverse seqs
    train_peaks = pd.concat([train_peaks, interim])

    #valid
    #change index to make it reverse
    new_index = valid_peaks.index.map(lambda x: x + "_r")
    #copy of the original data frame
    interim = valid_peaks.copy()
    interim.index = new_index
    #new df with reverse seqs
    valid_peaks = pd.concat([valid_peaks, interim])
    ####################################

    ####################################
    #add reverse complement sequences 
    #to train and valid
    ####################################
    #train
    train_sequences.index = train_sequences.index.map(str)
    new_index = train_sequences.index.map(lambda x: x + "_r")
    interim = train_sequences.copy()
    interim.index = new_index
    interim = interim.map(lambda x: Seq.reverse_complement(x))
    train_sequences = train_sequences.append(interim) 

    #valid
    valid_sequences.index = valid_sequences.index.map(str)
    new_index = valid_sequences.index.map(lambda x: x + "_r")
    interim = valid_sequences.copy()
    interim.index = new_index
    interim = interim.map(lambda x: Seq.reverse_complement(x))
    valid_sequences = valid_sequences.append(interim) 
    print("Added reverse complement sequences to train and valid data sets")
    
    print('Now %d of train sequences ' % train_peaks.shape[0])
    print('Now %d of validation sequences ' % valid_peaks.shape[0])
    ####################################

####################################
#one hot encoding
####################################
train_sequences = train_sequences.map(lambda x: dna_one_hot(x, flatten=False))
train_sequences = np.stack(train_sequences, axis=0)

valid_sequences = valid_sequences.map(lambda x: dna_one_hot(x, flatten=False))
valid_sequences = np.stack(valid_sequences, axis=0)

test_sequences = test_sequences.map(lambda x: dna_one_hot(x, flatten=False))
test_sequences = np.stack(test_sequences, axis=0)
print("Done encoding the data")
####################################

####################################
#saving the data
####################################
train_peaks = train_peaks.values
valid_peaks = valid_peaks.values
test_peaks = test_peaks.values

h5f = h5py.File(Output_h5, 'w')

h5f.create_dataset('target_labels', data=target_labels)

h5f.create_dataset('train_in', data=train_sequences)
h5f.create_dataset('train_out', data=train_peaks)
h5f.create_dataset('train_headers', data=train_headers)

h5f.create_dataset('valid_in', data=valid_sequences)
h5f.create_dataset('valid_out', data=valid_peaks)
h5f.create_dataset('valid_headers', data=valid_headers)

h5f.create_dataset('test_in', data=test_sequences)
h5f.create_dataset('test_out', data=test_peaks)

h5f.close()
print("Data saved, you are good to go")
