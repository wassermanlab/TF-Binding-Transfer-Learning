import plotly.graph_objects as go
import plotly.figure_factory as ff
from Bio.SeqUtils import GC
import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle
import sys
import h5py
import tqdm
import gzip


Matrix_file = sys.argv[1] #"../data/matrices/matrix2d.ReMap+UniBind.sparse.npz"
Multimodel_h5_file = sys.argv[2] #"../data/tf_peaks_50_partial.h5"
Regions_file = sys.argv[3] #"../data/idx_files/regions_idx.pickle.gz"
TF_file = sys.argv[4] #"../data/idx_files/tfs_idx.pickle.gz"
Output_new_final_df = sys.argv[5]

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
pwm_tfs.index = pwm_tfs.index.map(str)

data = h5py.File(Multimodel_h5_file, 'r')

#get headers that were used for training/validation
train_headers = list(data['train_headers'])
valid_headers = list(data['valid_headers'])

train_headers = [i.decode("utf-8") for i in train_headers]
valid_headers = [i.decode("utf-8") for i in valid_headers]
to_delete = train_headers + valid_headers

print('The shape used to be (%d, %d)' % (pwm_tfs.shape[0], pwm_tfs.shape[1]))
#delete sequences from the matrix
pwm_tfs = pwm_tfs.drop(to_delete)

print('Now the shape is (%d, %d)' % (pwm_tfs.shape[0], pwm_tfs.shape[1]))
print('We deleted %d sequences' % len(to_delete))

with open(Output_new_final_df, 'wb') as f:
    pickle.dump(pwm_tfs, f)
