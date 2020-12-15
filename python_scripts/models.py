import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch import relu, sigmoid
from collections import OrderedDict
import torch.nn.modules.activation as activation

#the deep learning model (Basset architecture)
class ConvNetDeep(nn.Module):
    def __init__(self, num_classes, weight_path=None) :
        super(ConvNetDeep, self).__init__()
        # Block 1 :
        self.c1 = nn.Conv1d(4,100,19)
        self.bn1 = nn.BatchNorm1d(100)
        self.rl1 = activation.ReLU() 
        self.mp1 = nn.MaxPool1d(3,3)

        # Block 2 :
        self.c2 = nn.Conv1d(100,200,7)
        self.bn2 = nn.BatchNorm1d(200)
        self.rl2 = activation.ReLU()
        self.mp2 = nn.MaxPool1d(3,3)

        # Block 3 :
        self.c3 = nn.Conv1d(200,200,4)
        self.bn3 = nn.BatchNorm1d(200)
        self.rl3 = activation.ReLU()
        self.mp3 = nn.MaxPool1d(3,3)

        # Block 4 : Fully Connected 1 :
        self.d4 = nn.Linear(1000,1000) #1000 for 200 input size
        self.bn4 = nn.BatchNorm1d(1000,1e-05,0.1,True)
        self.rl4 = activation.ReLU()
        self.dr4 =  nn.Dropout(0.3) 

        # Block 5 : Fully Connected 2 :
        self.d5 = nn.Linear(1000,1000)
        self.bn5 = nn.BatchNorm1d(1000,1e-05,0.1,True)
        self.rl5 = activation.ReLU()
        self.dr5 =  nn.Dropout(0.3) 

        # Block 6 :4Fully connected 3
        self.d6 = nn.Linear(1000,num_classes)
        #self.sig = activation.Sigmoid()
        
        if weight_path :
            self.load_weights(weight_path)

    def forward(self, x, embeddings = False) :
        """
            :param: embeddings : if True forward return embeddings along with the output
        """
        # Block 1
        # x is of size - batch, 4, 200
        x = self.rl1(self.bn1(self.c1(x))) # output - batch, 100, 182
        
        # we save the activations of the first layer (interpretation)
        activations = x # batch, 100, 182
        x = self.mp1(x) # output - batch, 100, 60

        # Block 2
        # input is of size batch, 100, 60
        x = self.mp2(self.rl2(self.bn2(self.c2(x)))) #output - batch, 200, 18

        # Block 3
        # input is of size batch, 200, 18
        em = self.mp3(self.rl3(self.bn3(self.c3(x)))) #output - batch, 200, 5

        # Flatten
        o = torch.flatten(em, start_dim=1) #output - batch, 1000

        # FC1
        #input is of size - batch, 1000
        o = self.dr4(self.rl4(self.bn4(self.d4(o)))) #output - batch, 1000

        # FC2
        #input is of size - batch, 1000
        o = self.dr5(self.rl5(self.bn5(self.d5(o)))) #output - batch, 1000

        # FC3
        #input is of size - batch, 1000
        #o = self.sig(self.d6(o)) #output - batch, num_of_classes
        o = self.d6(o) #doing BCEWithLogits #output - batch, num_of_classes
        
        #maximum for every filter (and corresponding index)
        activations, act_index = torch.max(activations, dim=2)

        if embeddings : return(o, activations, act_index, em)
        #return (o, activations, act_index)
        return o


    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1 :
                if v.shape[-1] ==1 :
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)
        
        
class ConvNetDeepBNAfterRelu(nn.Module):
    def __init__(self, num_classes, weight_path=None) :
        super(ConvNetDeepBNAfterRelu, self).__init__()
        # Block 1 :
        self.c1 = nn.Conv1d(4,100,19)
        self.bn1 = nn.BatchNorm1d(100)
        self.rl1 = activation.ReLU() 
        self.mp1 = nn.MaxPool1d(3,3)

        # Block 2 :
        self.c2 = nn.Conv1d(100,200,7)
        self.bn2 = nn.BatchNorm1d(200)
        self.rl2 = activation.ReLU()
        self.mp2 = nn.MaxPool1d(3,3)

        # Block 3 :
        self.c3 = nn.Conv1d(200,200,4)
        self.bn3 = nn.BatchNorm1d(200)
        self.rl3 = activation.ReLU()
        self.mp3 = nn.MaxPool1d(3,3)

        # Block 4 : Fully Connected 1 :
        self.d4 = nn.Linear(1000,1000) #1000 for 200 input size
        self.bn4 = nn.BatchNorm1d(1000,1e-05,0.1,True)
        self.rl4 = activation.ReLU()
        self.dr4 =  nn.Dropout(0.3) 

        # Block 5 : Fully Connected 2 :
        self.d5 = nn.Linear(1000,1000)
        self.bn5 = nn.BatchNorm1d(1000,1e-05,0.1,True)
        self.rl5 = activation.ReLU()
        self.dr5 =  nn.Dropout(0.3) 

        # Block 6 :4Fully connected 3
        self.d6 = nn.Linear(1000,num_classes)
        #self.sig = activation.Sigmoid()
        
        if weight_path :
            self.load_weights(weight_path)

    def forward(self, x, embeddings = False) :
        """
            :param: embeddings : if True forward return embeddings along with the output
        """
        # Block 1
        # x is of size - batch, 4, 200
        #x = self.rl1(self.bn1(self.c1(x))) # output - batch, 100, 182
        x = self.bn1(self.rl1(self.c1(x)))
        
        # we save the activations of the first layer (interpretation)
        activations = x # batch, 100, 182
        x = self.mp1(x) # output - batch, 100, 60

        # Block 2
        # input is of size batch, 100, 60
        x = self.mp2(self.bn2(self.rl2(self.c2(x)))) #output - batch, 200, 18

        # Block 3
        # input is of size batch, 200, 18
        em = self.mp3(self.bn3(self.rl3(self.c3(x)))) #output - batch, 200, 5

        # Flatten
        o = torch.flatten(em, start_dim=1) #output - batch, 1000

        # FC1
        #input is of size - batch, 1000
        o = self.dr4(self.bn4(self.rl4(self.d4(o)))) #output - batch, 1000

        # FC2
        #input is of size - batch, 1000
        o = self.dr5(self.bn5(self.rl5(self.d5(o)))) #output - batch, 1000

        # FC3
        #input is of size - batch, 1000
        #o = self.sig(self.d6(o)) #output - batch, num_of_classes
        o = self.d6(o) #doing BCEWithLogits #output - batch, num_of_classes
        
        #maximum for every filter (and corresponding index)
        activations, act_index = torch.max(activations, dim=2)

        if embeddings : return(o, activations, act_index, em)
        #return (o, activations, act_index)
        return o


    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1 :
                if v.shape[-1] ==1 :
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)
        
        
#the deep learning model (the shallow architecture)
class ConvNetShallow(nn.Module):
    def __init__(self, num_classes, weight_path=None) :
        super(ConvNetShallow, self).__init__()
        # Block 1 :
        self.c1 = nn.Conv1d(4,100,19) 
        self.bn1 = nn.BatchNorm1d(100)
        self.rl1 = activation.ReLU() 
        self.mp1 = nn.MaxPool1d(10,10)

        # Block 2 :
        self.c2 = nn.Conv1d(100,200,5)
        self.bn2 = nn.BatchNorm1d(200)
        self.rl2 = activation.ReLU()
        self.mp2 = nn.MaxPool1d(4,4)

        # Block 3 : Fully Connected 1 :
        self.d3 = nn.Linear(600,600) #600 for 200 input size
        self.bn3 = nn.BatchNorm1d(600,1e-05,0.1,True)
        self.rl3 = activation.ReLU()
        self.dr3 =  nn.Dropout(0.3) 

        # Block 4 :4Fully connected 3
        self.d4 = nn.Linear(600,num_classes)
        #self.sig = activation.Sigmoid()
        
        if weight_path :
            self.load_weights(weight_path)

    def forward(self, x, embeddings = False) :
        """
            :param: embeddings : if True forward return embeddings along with the output
        """
        # Block 1
        # x is of size - batch, 4, 200
        x = self.rl1(self.bn1(self.c1(x))) # output - batch, 100, 182
        
        # we save the activations of the first layer (interpretation)
        activations = x # batch, 100, 182
        x = self.mp1(x) # output - batch, 100, 18

        # Block 2
        # input is of size batch, 100, 18
        em = self.mp2(self.rl2(self.bn2(self.c2(x)))) #output - batch, 200, 3

        # Flatten
        o = torch.flatten(em, start_dim=1) #output - batch, 600

        # FC1
        #input is of size - batch, 1000
        o = self.dr3(self.rl3(self.bn3(self.d3(o)))) #output - batch, 600

        # FC2
        #input is of size - batch, 1000
        #o = self.sig(self.d6(o)) #output - batch, num_of_classes
        o = self.d4(o) #doing BCEWithLogits #output - batch, num_of_classes
        
        #maximum for every filter (and corresponding index)
        activations, act_index = torch.max(activations, dim=2)

        if embeddings : return(o, activations, act_index, em)
        #return (o, activations, act_index)
        return o


    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1 :
                if v.shape[-1] ==1 :
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)

        
class motifCNN(nn.Module):
    def __init__(self, original_model, num_classes):
        super(motifCNN, self).__init__()
        
        self.num_classes = num_classes
        
        self.c1 = original_model.c1
        self.bn1 = original_model.bn1
        self.rl1 = original_model.rl1
        self.mp1 = original_model.mp1

        # Block 2 :
        self.c2 = original_model.c2
        self.bn2 = original_model.bn2
        self.rl2 = original_model.rl2
        self.mp2 = original_model.mp2

        # Block 3 :
        self.c3 = original_model.c3
        self.bn3 = original_model.bn3
        self.rl3 = original_model.rl3
        self.mp3 = original_model.mp3

        # Block 4 : Fully Connected 1 :
        self.d4 = original_model.d4
        self.bn4 = original_model.bn4
        self.rl4 = original_model.rl4
        self.dr4 = original_model.dr4

        # Block 5 : Fully Connected 2 :
        self.d5 = original_model.d5
        self.bn5 = original_model.bn5
        self.rl5 = original_model.rl5
        self.dr5 = original_model.dr5

        # Block 6 :4Fully connected 3
        self.d6 = original_model.d6

    def forward(self, input):
        
        # Block 1
        x = self.rl1(self.bn1(self.c1(input))) 
        
        #we save the activations of the first layer
        layer1_activations = x 
        
        #do maxpooling for layer 1
        layer1_out = self.mp1(x) #batch, 100, 60

        #calculate average activation by filter for the whole batch
        filter_means_batch = layer1_activations.mean(0).mean(1) #[100]
        
        # run all other layers with 1 filter left out at a time
        batch_size = layer1_out.shape[0]
        filter_size = layer1_out.shape[1]
        last_dimension = layer1_out.shape[2] #60
        predictions = torch.zeros(batch_size, filter_size,  self.num_classes) #5

        for i in range(filter_size):
            #modify filter i of first layer output
            filter_input = layer1_out.clone()

            filter_input[:,i,:] = filter_input.new_full((batch_size, last_dimension), 
                                                        fill_value=filter_means_batch[i].item())

            # Block 2
            x = self.mp2(self.rl2(self.bn2(self.c2(filter_input)))) 

            # Block 3
            em = self.mp3(self.rl3(self.bn3(self.c3(x)))) 

            # Flatten
            o = torch.flatten(em, start_dim=1) 

            # FC1
            o = self.dr4(self.rl4(self.bn4(self.d4(o))))

            # FC2
            o = self.dr5(self.rl5(self.bn5(self.d5(o))))

            # FC3
            #o = self.sig(self.d6(o))
            o = self.d6(o) #doing BCEWithLogits

            predictions[:,i,:] = o
            
            activations, act_index = torch.max(layer1_activations, dim=2)

        return predictions, layer1_activations
    
    
class motifCNNBNAfterRelu(nn.Module):
    def __init__(self, original_model, num_classes):
        super(motifCNNBNAfterRelu, self).__init__()
        
        self.num_classes = num_classes
        
        self.c1 = original_model.c1
        self.bn1 = original_model.bn1
        self.rl1 = original_model.rl1
        self.mp1 = original_model.mp1

        # Block 2 :
        self.c2 = original_model.c2
        self.bn2 = original_model.bn2
        self.rl2 = original_model.rl2
        self.mp2 = original_model.mp2

        # Block 3 :
        self.c3 = original_model.c3
        self.bn3 = original_model.bn3
        self.rl3 = original_model.rl3
        self.mp3 = original_model.mp3

        # Block 4 : Fully Connected 1 :
        self.d4 = original_model.d4
        self.bn4 = original_model.bn4
        self.rl4 = original_model.rl4
        self.dr4 = original_model.dr4

        # Block 5 : Fully Connected 2 :
        self.d5 = original_model.d5
        self.bn5 = original_model.bn5
        self.rl5 = original_model.rl5
        self.dr5 = original_model.dr5

        # Block 6 :4Fully connected 3
        self.d6 = original_model.d6

    def forward(self, input):
        
        # Block 1
        #x = self.rl1(self.bn1(self.c1(input))) 
        x = self.bn1(self.rl1(self.c1(input)))
        
        #we save the activations of the first layer
        layer1_activations = x 
        
        #do maxpooling for layer 1
        layer1_out = self.mp1(x) #batch, 100, 60

        #calculate average activation by filter for the whole batch
        filter_means_batch = layer1_activations.mean(0).mean(1) #[100]
        
        # run all other layers with 1 filter left out at a time
        batch_size = layer1_out.shape[0]
        filter_size = layer1_out.shape[1]
        last_dimension = layer1_out.shape[2] #60
        predictions = torch.zeros(batch_size, filter_size,  self.num_classes) #5

        for i in range(filter_size):
            #modify filter i of first layer output
            filter_input = layer1_out.clone()

            filter_input[:,i,:] = filter_input.new_full((batch_size, last_dimension), 
                                                        fill_value=filter_means_batch[i].item())

            # Block 2
            x = self.mp2(self.bn2(self.rl2(self.c2(filter_input)))) 

            # Block 3
            em = self.mp3(self.bn3(self.rl3(self.c3(x)))) 

            # Flatten
            o = torch.flatten(em, start_dim=1) 

            # FC1
            o = self.dr4(self.bn4(self.rl4(self.d4(o))))

            # FC2
            o = self.dr5(self.bn5(self.rl5(self.d5(o))))

            # FC3
            #o = self.sig(self.d6(o))
            o = self.d6(o) #doing BCEWithLogits

            predictions[:,i,:] = o
            
            activations, act_index = torch.max(layer1_activations, dim=2)

        return predictions, layer1_activations
    
    
#DanQ pytorch implementation 
#taken from - https://github.com/PuYuQian/PyDanQ/blob/master/DanQ_train.py
class DanQ(nn.Module):
    def __init__(self, num_classes, weight_path=None):
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(13*640, 925)
        self.Linear2 = nn.Linear(925, num_classes)
        
        if weight_path :
            self.load_weights(weight_path)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n,h_c) = self.BiLSTM(x_x)
        x = x.contiguous().view(-1, 13*640)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x
    
    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1 :
                if v.shape[-1] ==1 :
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)
    
    
#with batch size 50
#for seqs, labels in dataloaders["train"]:
#    print(seqs.shape) #torch.Size([50, 4, 200])
#    x = mp1(rl1(bn1(c1(seqs))))
#    print(x.shape) #torch.Size([50, 320, 13])
#    x_x = torch.transpose(x, 1, 2)
#    print(x_x.shape) #torch.Size([50, 13, 320])
#    x, (h_n,h_c) = BiLSTM(x_x)
#    print(x.shape) #torch.Size([50, 13, 640])
#    x = x.contiguous().view(-1, 13*640) 
#    print(x.shape) #torch.Size([50, 8320])
#    x = Linear1(x)
#    print(x.shape) #torch.Size([50, 925])
#    x = F.relu(x)
#    print(x.shape) #torch.Size([50, 925])
#    x = Linear2(x)
#    print(x.shape) #torch.Size([50, 50])
#    break