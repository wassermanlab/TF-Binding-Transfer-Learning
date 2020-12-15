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

Matrix_pkl_file_1 = sys.argv[1] #example - 'new_data/tf_peaks_50_noNs.pkl'
Matrix_pkl_file_2 = sys.argv[2] 
Seed = sys.argv[3] #seed for permutations; example - 1
Fasta_pkl_file = sys.argv[4] #example - new_data/fasta_sequences_50.pkl
test_pct = sys.argv[5] #fraction of test data; example - 0.1 (it means 10%)
valid_pct = sys.argv[6] #fraction of validation data; example - 0.1 (it means 10%)
Output_h5_1 = sys.argv[7] #example - my_own_h5_50.h5
Output_h5_2 = sys.argv[8] #example - my_own_h5_50.h5
Add_reverse_complement = sys.argv[9] #Add reverse complement or not - True/False

with open(Matrix_pkl_file_1, 'rb') as f:
    tf_peaks_1 = pickle.load(f)
    
with open(Matrix_pkl_file_2, 'rb') as f:
    tf_peaks_2 = pickle.load(f)

with open(Fasta_pkl_file, 'rb') as f:
    fasta_sequences = pickle.load(f)

####################################
#shuffle the sequence
####################################
npr.seed(int(Seed)) #setting a seed

order = npr.permutation(tf_peaks_1.index) #indexes are the same
tf_peaks_1 = tf_peaks_1.reindex(order) #labels
tf_peaks_2 = tf_peaks_2.reindex(order)
fasta_sequences = fasta_sequences[order] #sequences
print("Done shuffling")
####################################

test_count = int(0.5 + float(test_pct) * tf_peaks_1.shape[0])
valid_count = int(0.5 + float(valid_pct) * tf_peaks_1.shape[0])
train_count = tf_peaks_1.shape[0] - test_count - valid_count

print('%d training sequences ' % train_count)
print('%d test sequences ' % test_count)
print('%d validation sequences ' % valid_count)

####################################
#shuffle the sequence
####################################
i = 0
train_peaks_1, train_sequences = tf_peaks_1.iloc[i:i+train_count,:], fasta_sequences[i:i+train_count]
i += train_count
valid_peaks_1, valid_sequences = tf_peaks_1.iloc[i:i+valid_count,:], fasta_sequences[i:i+valid_count]
i += valid_count
test_peaks_1, test_sequences = tf_peaks_1.iloc[i:i+test_count,:], fasta_sequences[i:i+test_count]

i = 0
train_peaks_2 = tf_peaks_2.iloc[i:i+train_count,:]
i += train_count
valid_peaks_2 = tf_peaks_2.iloc[i:i+valid_count,:]
i += valid_count
test_peaks_2 = tf_peaks_2.iloc[i:i+test_count,:]
####################################

####################################
#saving TF labels and sequences ids
#which we used for training/validation
####################################
target_labels_1 = list(tf_peaks_1)
train_peaks_1.index = train_peaks_1.index.map(str)
valid_peaks_1.index = valid_peaks_1.index.map(str)
test_peaks_1.index = test_peaks_1.index.map(str)

train_headers_1 = train_peaks_1.index
valid_headers_1 = valid_peaks_1.index

target_labels_1 = [a.encode('utf8') for a in target_labels_1]
train_headers_1 = [a.encode('utf8') for a in train_headers_1]
valid_headers_1 = [a.encode('utf8') for a in valid_headers_1]

###

target_labels_2 = list(tf_peaks_2)
train_peaks_2.index = train_peaks_2.index.map(str)
valid_peaks_2.index = valid_peaks_2.index.map(str)
test_peaks_2.index = test_peaks_2.index.map(str)

train_headers_2 = train_peaks_2.index
valid_headers_2 = valid_peaks_2.index

target_labels_2 = [a.encode('utf8') for a in target_labels_2]
train_headers_2 = [a.encode('utf8') for a in train_headers_2]
valid_headers_2 = [a.encode('utf8') for a in valid_headers_2]
####################################

if Add_reverse_complement == "True":
    ####################################
    #FIRST MATRIX
    ####################################
    #add reverse complement ids
    #to train and valid dfs
    ####################################
    #train
    #change index to make it reverse
    new_index_1 = train_peaks_1.index.map(lambda x: x + "_r")
    #copy of the original data frame
    interim_1 = train_peaks_1.copy()
    interim_1.index = new_index_1
    #new df with reverse seqs
    train_peaks_1 = pd.concat([train_peaks_1, interim_1])

    #valid
    #change index to make it reverse
    new_index_1 = valid_peaks_1.index.map(lambda x: x + "_r")
    #copy of the original data frame
    interim_1 = valid_peaks_1.copy()
    interim_1.index = new_index_1
    #new df with reverse seqs
    valid_peaks_1 = pd.concat([valid_peaks_1, interim_1])
    ####################################
    
    ####################################
    #SECOND MATRIX
    ####################################
    #add reverse complement ids
    #to train and valid dfs
    ####################################
    #train
    #change index to make it reverse
    new_index_2 = train_peaks_2.index.map(lambda x: x + "_r")
    #copy of the original data frame
    interim_2 = train_peaks_2.copy()
    interim_2.index = new_index_2
    #new df with reverse seqs
    train_peaks_2 = pd.concat([train_peaks_2, interim_2])

    #valid
    #change index to make it reverse
    new_index_2 = valid_peaks_2.index.map(lambda x: x + "_r")
    #copy of the original data frame
    interim_2 = valid_peaks_2.copy()
    interim_2.index = new_index_2
    #new df with reverse seqs
    valid_peaks_2 = pd.concat([valid_peaks_2, interim_2])
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
    
    print('Now %d of train sequences in 50 ' % train_peaks_1.shape[0])
    print('Now %d of validation sequences in 50 ' % valid_peaks_1.shape[0])
    
    print('Now %d of train sequences in 5 ' % train_peaks_2.shape[0])
    print('Now %d of validation sequences in 5 ' % valid_peaks_2.shape[0])
    
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
#FIRST MATRIX
####################################
train_peaks_1 = train_peaks_1.values
valid_peaks_1 = valid_peaks_1.values
test_peaks_1 = test_peaks_1.values

h5f = h5py.File(Output_h5_1, 'w')

h5f.create_dataset('target_labels', data=target_labels_1)

h5f.create_dataset('train_in', data=train_sequences)
h5f.create_dataset('train_out', data=train_peaks_1)
h5f.create_dataset('train_headers', data=train_headers_1)

h5f.create_dataset('valid_in', data=valid_sequences)
h5f.create_dataset('valid_out', data=valid_peaks_1)
h5f.create_dataset('valid_headers', data=valid_headers_1)

h5f.create_dataset('test_in', data=test_sequences)
h5f.create_dataset('test_out', data=test_peaks_1)

h5f.close()
print("Data saved for 50 TFs, you are good to go")

####################################
#saving the data
#SECOND MATRIX
####################################
train_peaks_2 = train_peaks_2.values
valid_peaks_2 = valid_peaks_2.values
test_peaks_2 = test_peaks_2.values

h5f = h5py.File(Output_h5_2, 'w')

h5f.create_dataset('target_labels', data=target_labels_2)

h5f.create_dataset('train_in', data=train_sequences)
h5f.create_dataset('train_out', data=train_peaks_2)
h5f.create_dataset('train_headers', data=train_headers_2)

h5f.create_dataset('valid_in', data=valid_sequences)
h5f.create_dataset('valid_out', data=valid_peaks_2)
h5f.create_dataset('valid_headers', data=valid_headers_2)

h5f.create_dataset('test_in', data=test_sequences)
h5f.create_dataset('test_out', data=test_peaks_2)

h5f.close()
print("Data saved for 5 TFs, you are good to go")