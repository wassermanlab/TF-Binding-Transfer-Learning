from Bio.SeqUtils import GC
import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle
import sys

TF_name = sys.argv[1] #"CTCF" 
Matrix_pkl_file = sys.argv[2] #"../data/final_df_dropped.pkl"
Fasta_sequences = sys.argv[3] #"../data/sequences/sequences.200bp.fa"
Subsample_number = sys.argv[4] #10000
Test_number = sys.argv[5] #5000
Output_fasta_file = sys.argv[6] #Fasta file with the sequences for training (new_data/fasta_sequences_CTCF.pkl)
Output_act_file = sys.argv[7] #Act file with the sequences for training (new_data/tf_peaks_CTCF.pkl)
Output_fasta_test_file = sys.argv[8]
Output_act_test_file = sys.argv[9]

pkl_file = open(Matrix_pkl_file, 'rb')
pwm_tfs = pickle.load(pkl_file)
pkl_file.close()

#get peaks only for the provided TF
peaks = pwm_tfs[TF_name].dropna()

#none zero peaks
nonzero = peaks[np.where(peaks == 1)[0]].index
#zero peaks
zero = peaks[np.where(peaks == 0)[0]].index

fasta_ids_nonzero = {}
fasta_ids_zero = {}

Subsample_number = int(Subsample_number)

ratio_to_subsample = Subsample_number/nonzero.shape[0]

#get fasta sequences
fasta_sequences = SeqIO.parse(open(Fasta_sequences),'fasta')
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)

    name = name.split(":")[0]

    if str(name) in nonzero:
        #delete sequences with Ns
        if "N" not in sequence.upper():
            fasta_ids_nonzero[name] = sequence.upper()
    elif str(name) in zero:
        #delete sequences with Ns
        if "N" not in sequence.upper():
            fasta_ids_zero[name] = sequence.upper()
                        
fasta_ids_nonzero = pd.Series(fasta_ids_nonzero)
fasta_ids_zero = pd.Series(fasta_ids_zero)

#calculate the GC composition for the extracted sequences
nonzero_gc = fasta_ids_nonzero.apply(lambda x: GC(x.upper()))
zero_gc = fasta_ids_zero.apply(lambda x: GC(x.upper()))

bins = [0,10,20,30,40,50,60,70,80,90,100]
labels = [10,20,30,40,50,60,70,80,90,100]

#assigning bins from nonzero
binned_nonzero = pd.cut(nonzero_gc, bins = bins,
               labels = labels)

#assigning bins from zero
binned_zero = pd.cut(zero_gc, bins = bins,
               labels = labels)

#sampling new ones
new_nonzero_ind = []
for l in labels:
    num_nonzero = len(binned_nonzero[binned_nonzero == l])
    
    #if there are no nonzero peaks, continue
    if num_nonzero == 0:
        continue

    sampled_bins = binned_nonzero[binned_nonzero == l].sample(n=int(num_nonzero*ratio_to_subsample),
                                                              replace=False)
    new_nonzero_ind = new_nonzero_ind + list(sampled_bins.index)
    
fasta_new_ids_nonzero = fasta_ids_nonzero[new_nonzero_ind]
new_nonzero_gc = fasta_new_ids_nonzero.apply(lambda x: GC(x.upper()))

binned_new_nonzero = binned_nonzero[new_nonzero_ind]

#sampling new zeros
new_zero_ind = []
for l in labels:
    num_nonzero = len(binned_new_nonzero[binned_new_nonzero == l])
    num_zero = len(binned_zero[binned_zero == l])

    #if there are no nonzero peaks, continue
    if num_nonzero == 0 or num_zero == 0:
        continue

    if num_zero >= num_nonzero:
        #sample without replacement
        sampled_bins = binned_zero[binned_zero == l].sample(n=num_nonzero, replace=False)
        new_zero_ind = new_zero_ind + list(sampled_bins.index)

    if num_nonzero > num_zero:
        print("For bin %s we have more nonzeros than zeros!" % l)
        sampled_bins = binned_zero[binned_zero == l]
        new_zero_ind = new_zero_ind + list(sampled_bins.index)
        
fasta_new_ids_zero = fasta_ids_zero[new_zero_ind]
new_zero_gc = fasta_new_ids_zero.apply(lambda x: GC(x.upper()))

#saving files
new_zero_label = pd.Series(np.zeros(len(new_zero_gc)), index = new_zero_gc.index).astype(int)
new_nonzero_label = pd.Series(np.ones(len(new_nonzero_gc)), index = new_nonzero_gc.index).astype(int)

labels = new_zero_label.append(new_nonzero_label)
df = labels.to_frame()
df = df.rename(columns= {0: TF_name})

fasta_sequences = fasta_new_ids_zero.append(fasta_new_ids_nonzero)

#save the matrix
with open(Output_act_file, 'wb') as f:
    pickle.dump(df, f)

#save the fasta sequences
with open(Output_fasta_file, 'wb') as f:
    pickle.dump(fasta_sequences, f)
    
print("Subsampled for training")
#########################################################################
#Subsample for test
filtered_peaks = peaks.drop(fasta_sequences.index)
#none zero peaks
filtered_nonzero = filtered_peaks[np.where(filtered_peaks == 1)[0]].index
#zero peaks
filtered_zero = filtered_peaks[np.where(filtered_peaks == 0)[0]].index

fasta_ids_nonzero = {}
fasta_ids_zero = {}

Test_number = int(Test_number)

ratio_to_subsample = Test_number/filtered_nonzero.shape[0]

#get fasta sequences
fasta_sequences = SeqIO.parse(open(Fasta_sequences),'fasta')
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)

    name = name.split(":")[0]

    if str(name) in filtered_nonzero:
        #delete sequences with Ns
        if "N" not in sequence.upper():
            fasta_ids_nonzero[name] = sequence.upper()
    elif str(name) in filtered_zero:
        #delete sequences with Ns
        if "N" not in sequence.upper():
            fasta_ids_zero[name] = sequence.upper()

fasta_ids_nonzero = pd.Series(fasta_ids_nonzero)
fasta_ids_zero = pd.Series(fasta_ids_zero)

#calculate the GC composition for the extracted sequences
nonzero_gc = fasta_ids_nonzero.apply(lambda x: GC(x.upper()))
zero_gc = fasta_ids_zero.apply(lambda x: GC(x.upper()))

bins = [0,10,20,30,40,50,60,70,80,90,100]
labels = [10,20,30,40,50,60,70,80,90,100]

#assigning bins from nonzero
binned_nonzero = pd.cut(nonzero_gc, bins = bins,
               labels = labels)

#assigning bins from zero
binned_zero = pd.cut(zero_gc, bins = bins,
               labels = labels)

#sampling new ones
new_nonzero_ind = []
for l in labels:
    num_nonzero = len(binned_nonzero[binned_nonzero == l])
    
    #if there are no nonzero peaks, continue
    if num_nonzero == 0:
        continue

    sampled_bins = binned_nonzero[binned_nonzero == l].sample(n=int(num_nonzero*ratio_to_subsample),
                                                              replace=False)
    new_nonzero_ind = new_nonzero_ind + list(sampled_bins.index)
    
fasta_new_ids_nonzero = fasta_ids_nonzero[new_nonzero_ind]
new_nonzero_gc = fasta_new_ids_nonzero.apply(lambda x: GC(x.upper()))

binned_new_nonzero = binned_nonzero[new_nonzero_ind]

#sampling new zeros
new_zero_ind = []
for l in labels:
    num_nonzero = len(binned_new_nonzero[binned_new_nonzero == l])
    num_zero = len(binned_zero[binned_zero == l])

    #if there are no nonzero peaks, continue
    if num_nonzero == 0 or num_zero == 0:
        continue

    if num_zero >= num_nonzero:
        #sample without replacement
        sampled_bins = binned_zero[binned_zero == l].sample(n=num_nonzero, replace=False)
        new_zero_ind = new_zero_ind + list(sampled_bins.index)

    if num_nonzero > num_zero:
        print("For bin %s we have more nonzeros than zeros!" % l)
        sampled_bins = binned_zero[binned_zero == l]
        new_zero_ind = new_zero_ind + list(sampled_bins.index)
        
fasta_new_ids_zero = fasta_ids_zero[new_zero_ind]
new_zero_gc = fasta_new_ids_zero.apply(lambda x: GC(x.upper()))

#saving files
new_zero_label = pd.Series(np.zeros(len(new_zero_gc)), index = new_zero_gc.index).astype(int)
new_nonzero_label = pd.Series(np.ones(len(new_nonzero_gc)), index = new_nonzero_gc.index).astype(int)

labels = new_zero_label.append(new_nonzero_label)
filtered_df = labels.to_frame()
filtered_df = filtered_df.rename(columns= {0: TF_name})

filtered_fasta_sequences = fasta_new_ids_zero.append(fasta_new_ids_nonzero)

#save the matrix
with open(Output_act_test_file, 'wb') as f:
    pickle.dump(filtered_df, f)

#save the fasta sequences
with open(Output_fasta_test_file, 'wb') as f:
    pickle.dump(filtered_fasta_sequences, f)
    
print("Subsampled for test")