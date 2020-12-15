from tools import *
from models import *
#import plotly.graph_objects as go
#import plotly.figure_factory as ff
from Bio.SeqUtils import GC
from Bio import SeqIO
import sys
import os
import pickle

#for reproducibility
seed = 42
np.random.seed(seed)

#correspondance
###############################################################################
H5_file = sys.argv[1] #the h5 file location (RELA_h5/tf_peaks_RELA.h5)
Output_weights = sys.argv[2] #folder where to store the weights (String_weights/RELA_real_model_weights)
Num_epochs = sys.argv[3] #number of epochs (50)
Batch_size = sys.argv[4] #batch size to use (100)
Learning_rate = sys.argv[5] #learning rate (0.003)
Output_labels = sys.argv[6]
###############################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = int(Num_epochs)
batch_size = int(Batch_size)
learning_rate = float(Learning_rate)

dataloaders, target_labels, train_out = load_datas(H5_file, batch_size)

target_labels = [i.decode("utf-8") for i in target_labels]

with open(Output_labels, 'wb') as f:
    pickle.dump(target_labels, f)

num_classes = len(target_labels) #number of classes

model = ConvNetDeep(num_classes).to(device)
#model = ConvNetDeepBNAfterRelu(num_classes).to(device)
#model = DanQ(num_classes).to(device)

criterion = nn.BCEWithLogitsLoss() #- no weights
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if not os.path.exists(Output_weights):
    os.makedirs(Output_weights)

model, train_error, test_error, train_fscore, test_fscore = train_model(dataloaders['train'],
                                                                        dataloaders['valid'], model, device, criterion,
                                                                        optimizer, num_epochs,
                                                                        Output_weights, "", 
                                                                        verbose=False) 

print("Done with training")
