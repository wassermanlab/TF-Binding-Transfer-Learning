from Bio import SeqIO
import os
import sys
import numpy as np
import pandas as pd
import pickle
import gzip
import random
import copy

######################################################################
TF_name = sys.argv[1] #"JUND"
Matrix_file = sys.argv[2] #"../data/matrices/matrix2d.ReMap+UniBind.partial.npz"
Regions_file = sys.argv[3] #"../data/idx_files/regions_idx.pickle.gz"
TF_file = sys.argv[4] #"../data/idx_files/tfs_idx.pickle.gz"
Cofactors = sys.argv[5] #"../data/cofactors.pickle"
All_peaks_fasta_file_path = sys.argv[6] #"../data/sequences/sequences.200bp.fa"
Output_folder = sys.argv[7]

Sampling_size = 40000 #(for bms, cofactors, string, best string)
#Sampling_size = 35000 #(only for best cor)
######################################################################

data = np.load(Matrix_file)

for i in data.files:
    matrix2d = data[i]
    
with gzip.open(Regions_file, 'rb') as f:
    regions = pickle.load(f)
    
with gzip.open(TF_file, 'rb') as f:
    tfs = pickle.load(f)
    
regions = pd.Series(regions).sort_values()
tfs = pd.Series(tfs).sort_values()

pwm_tfs = pd.DataFrame(matrix2d, columns = tfs.index, index = regions.values)

with open(Cofactors, 'rb') as f:
    cofactors = pickle.load(f)
    
######################################################################
#calculate the number of None regions per TF
nan_sums = pwm_tfs.isna().sum()

nan_perc = nan_sums/pwm_tfs.shape[0]*100
nan_perc = nan_perc.sort_values(ascending=True)

#building TF list
target_tf = TF_name
tf_same_bm = copy.deepcopy(cofactors[target_tf])

for i in nan_perc.index:
    if i not in tf_same_bm and i != target_tf:
        tf_same_bm.append(i)
    if len(tf_same_bm) == 50:
        break
        
tf_peaks_50 = pwm_tfs[tf_same_bm]
tf_peaks_50 = tf_peaks_50.dropna() 

print(tf_same_bm)

tf_same_bm = copy.deepcopy(cofactors[target_tf])

tf_peaks_5 = pwm_tfs[tf_same_bm]
tf_peaks_5 = tf_peaks_5.dropna() 

print(tf_same_bm)

sampled_regions = random.sample(list(tf_peaks_50.index), Sampling_size)

tf_peaks_50_sampled = tf_peaks_50.loc[sampled_regions,:]
tf_peaks_5_sampled = tf_peaks_5.loc[sampled_regions,:]
######################################################################

#save only TF related fasta sequences (and delete sequences with Ns)
fasta_sequences = SeqIO.parse(open(All_peaks_fasta_file_path),'fasta')

fasta_sequences_sampled = {}
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)

    #new format sequence file
    name = name.split(":")[0]

    if int(name) in sampled_regions:
        #remove sequences with Ns
        if "N" in sequence.upper():
            continue
        else:
            fasta_sequences_sampled[int(name)] = sequence.upper()

fasta_sequences_sampled = pd.Series(fasta_sequences_sampled)
         
tf_peaks_50_sampled_noNs = tf_peaks_50_sampled.loc[fasta_sequences_sampled.index,:]
tf_peaks_5_sampled_noNs = tf_peaks_5_sampled.loc[fasta_sequences_sampled.index,:]

######################################################################
#save the matrix
with open(Output_folder + "/tf_peaks_" + TF_name + "_50_act.pkl", 'wb') as f:
    pickle.dump(tf_peaks_50_sampled_noNs, f)
    
#save the matrix
with open(Output_folder + "/tf_peaks_" + TF_name + "_5_act.pkl", 'wb') as f:
    pickle.dump(tf_peaks_5_sampled_noNs, f)
    
#save the fasta sequences
with open(Output_folder + "/tf_peaks_" + TF_name + "_fasta.pkl", 'wb') as f:
    pickle.dump(fasta_sequences_sampled, f)
    
######################################################################

print("Done building fasta and act files for %s" % TF_name)