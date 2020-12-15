from Bio import SeqIO
import os
import sys
import numpy as np
import pandas as pd
import pickle
import gzip
import random
import copy

TF_name = sys.argv[1] #"JUND"
Matrix_file = sys.argv[2] #"../data/matrices/matrix2d.ReMap+UniBind.partial.npz"
Regions_file = sys.argv[3] #"../data/idx_files/regions_idx.pickle.gz"
TF_file = sys.argv[4] #"../data/idx_files/tfs_idx.pickle.gz"
Cofactors = sys.argv[5] #'../data/cofactors.pickle'
All_peaks_fasta_file_path = sys.argv[6] #"../data/sequences/sequences.200bp.fa"
Include_target_TF = sys.argv[7] #"True"
Output_folder = sys.argv[8]

Sampling_size = 70000

if not os.path.exists(Output_folder + "/"):
    os.makedirs(Output_folder + "/")
    
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
    
if Include_target_TF == "True":
    TFs_to_use = [TF_name]
    TFs_to_use = TFs_to_use + cofactors[TF_name][:-1]
else:
    TFs_to_use = cofactors[TF_name]
    
print("Use the following TFs: " + " ".join(TFs_to_use))
    
TF_peaks = pwm_tfs[TFs_to_use]
TF_peaks = TF_peaks.dropna()

shape = TF_peaks.shape[0]

print("We had " + str(shape) + " regions with cofactors")
if shape > Sampling_size:
    TF_peaks = TF_peaks.sample(n=Sampling_size, replace=False, axis=0)

print("Now we have " + str(TF_peaks.shape[0]) + " regions")

#save only TF related fasta sequences (and delete sequences with Ns)
fasta_sequences = SeqIO.parse(open(All_peaks_fasta_file_path),'fasta')

fasta_sequences_sampled = {}
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)

    #new format sequence file
    name = name.split(":")[0]

    if int(name) in TF_peaks.index:
        #remove sequences with Ns
        if "N" in sequence.upper():
            continue
        else:
            fasta_sequences_sampled[int(name)] = sequence.upper()

fasta_sequences_sampled = pd.Series(fasta_sequences_sampled)
         
TF_peaks_noNs = TF_peaks.loc[fasta_sequences_sampled.index,:]

#save the matrix
with open(Output_folder + "/tf_peaks_" + TF_name + "_act.pkl", 'wb') as f:
    pickle.dump(TF_peaks_noNs, f)
    
#save the fasta sequences
with open(Output_folder + "/tf_peaks_" + TF_name + "_fasta.pkl", 'wb') as f:
    pickle.dump(fasta_sequences_sampled, f)
    