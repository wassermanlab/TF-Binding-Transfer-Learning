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
Number_of_partners = sys.argv[5] #5
Cluster_modes = sys.argv[6] #'../data/clusters_multi_modes_sorted.pickle'
Cluster_corr = sys.argv[7] #'../data/tf_clust_corr.pickle'
All_peaks_fasta_file_path = sys.argv[8] #"../data/sequences/sequences.200bp.fa"
BM = sys.argv[9] #"1"
Real = sys.argv[10] #"True"
Include_target_TF = sys.argv[11] #"True"
Output_folder = sys.argv[12]

if not os.path.exists(Output_folder + "/"):
    os.makedirs(Output_folder + "/")

BM = str(BM)
Number_of_partners = int(Number_of_partners)

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

with open(Cluster_modes, 'rb') as f:
    clusters_multi_modes_sorted = pickle.load(f)
    
with open(Cluster_corr, 'rb') as f:
    tf_clust_corr = pickle.load(f)
    


###################################################################################
if Real == "True":

    tfs = copy.deepcopy(clusters_multi_modes_sorted[BM])

    if Include_target_TF == "True":

        sampled_tfs = [TF_name]
        tfs.remove(TF_name)
        sampled_tfs_2 = random.sample(tfs, Number_of_partners-1)
        sampled_tfs = sampled_tfs + sampled_tfs_2

    else:
        tfs.remove(TF_name)
        sampled_tfs = random.sample(tfs, Number_of_partners)

else:

    bms_to_avoid = tf_clust_corr[TF_name].split(",")
    new_clusters = clusters_multi_modes_sorted.drop(bms_to_avoid)
    remaining_tfs =[]
    for x in new_clusters:
        remaining_tfs.extend(x)
    remaining_tfs = list(set(remaining_tfs))

    if Include_target_TF == "True":

        sampled_tfs = [TF_name]
        if TF_name in remaining_tfs:
            remaining_tfs.remove(TF_name)
        sampled_tfs_2 = random.sample(remaining_tfs, Number_of_partners-1)
        sampled_tfs = sampled_tfs + sampled_tfs_2

    else:
        if TF_name in remaining_tfs:
            remaining_tfs.remove(TF_name)
        sampled_tfs = random.sample(remaining_tfs, Number_of_partners)

TF_peaks = pwm_tfs[sampled_tfs]
TF_peaks = TF_peaks.dropna()

print("Sampled the following TFs")
print(sampled_tfs)
###################################################################################

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
    

print("Done building fasta and act files for %s" % TF_name)
