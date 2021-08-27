from tools import *
from models import *
from collections import OrderedDict
import pickle
from captum.attr import (
    GuidedGradCam,

    DeepLift,

    IntegratedGradients
)

############ demonstrating the usage of DeepLIFT attributions on JUND sequences ######################


JUND_index = 0  #  0 for individual model, else corresponds to output node number of JUND in multi-model

# create dictionary of importance scores and hypothetical importance scores
# these dictionaries are used as input for running TFModisco
# the keys of the dictionary are the tasks of our interest - in this case JUND

task_to_scores = OrderedDict()  # dictionary of importance scores
task_to_scores["JUND"] = []

task_to_hyp_scores = OrderedDict() # dictionary hypothetical importance scores
task_to_hyp_scores["JUND"] = []

one_hot = []  # to store one hot vectors of sequences , used as input for tfMODISCO

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# loading data

dataloaders, target_labels, train_out = load_datas("JUND.h5", 1000)

batch_num = 0
for seq, label in dataloaders['test']:

    seq = seq.float().to(device)

    batch_num = batch_num + 1



    num_classes = 1  # number of output classes , in case of individual model = 1

    model = ConvNetDeep(num_classes).to(device)

    model.load_state_dict(torch.load(
        "Model_weights_random/JUND_real_indiv_weights_1/JUND_tl_weights/model_epoch_3_.pth"))  # change the path to load model weights
    model.eval()
    sigmoid = nn.Sigmoid()

    model_preds = model(seq)
    model_preds = sigmoid(model_preds)

    # UNCOMMENT WHEN USING MULTIMODEL

    # model_preds = model_preds[:,JUND_index]  # select predictions corresponding to JUND node

    idx = torch.nonzero(model_preds > 0.5)  # selecting indices where predictions are greater than 0.5

    idx = idx[:, 0]

    seq = seq[idx, :, :]  # selecting sequences where prediction > 0.5

    # creating hypothetical sequences - A's at all position, C's at all positions and so on

    hypo_A = np.array([np.array([np.array([1, 0, 0, 0]) for i in range(seq.shape[2])]).T for j in range(seq.shape[0])])
    hypo_A = torch.from_numpy(hypo_A).float().to(device)

    hypo_C = np.array([np.array([np.array([0, 1, 0, 0]) for i in range(seq.shape[2])]).T for j in range(seq.shape[0])])
    hypo_C = torch.from_numpy(hypo_C).float().to(device)

    hypo_G = np.array([np.array([np.array([0, 0, 1, 0]) for i in range(seq.shape[2])]).T for j in range(seq.shape[0])])
    hypo_G = torch.from_numpy(hypo_G).float().to(device)

    hypo_T = np.array([np.array([np.array([0, 0, 0, 1]) for i in range(seq.shape[2])]).T for j in range(seq.shape[0])])
    hypo_T = torch.from_numpy(hypo_T).float().to(device)

    number_references = 10  # imp scores are averaged over 10 reference sequences
    hypo_A_scores = []
    hypo_C_scores = []
    hypo_G_scores = []
    hypo_T_scores = []
    importance_scores = []

    for ref in range(number_references):
        seq_ref = seq[:, :, torch.randperm(seq.size()[2])]

        # obtaining attribution scores

        dl = DeepLift(model, multiply_by_inputs=False)
        attributions, delta = dl.attribute(seq, seq_ref, target=JUND_index, return_convergence_delta=True)

        attribution_imp = (seq - seq_ref) * attributions  # obtaining importance scores

        imp_score = torch.sum(attribution_imp, axis=1)  # summing across all 4 bases
        importance_scores.append(imp_score)

        # obtaining hypothetical importance scores for A,C,G,T

        attributions_hypo_A = (hypo_A - seq_ref) * attributions
        attributions_A = torch.sum(attributions_hypo_A, axis=1)
        hypo_A_scores.append(attributions_A)

        attributions_hypo_C = (hypo_C - seq_ref) * attributions
        attributions_C = torch.sum(attributions_hypo_C, axis=1)
        hypo_C_scores.append(attributions_C)

        attributions_hypo_G = (hypo_G - seq_ref) * attributions
        attributions_G = torch.sum(attributions_hypo_G, axis=1)
        hypo_G_scores.append(attributions_G)

        attributions_hypo_T = (hypo_T - seq_ref) * attributions
        attributions_T = torch.sum(attributions_hypo_T, axis=1)
        hypo_T_scores.append(attributions_T)

    hypo_A_scores = torch.stack(hypo_A_scores, dim=0).mean(dim=0)
    hypo_C_scores = torch.stack(hypo_C_scores, dim=0).mean(dim=0)
    hypo_G_scores = torch.stack(hypo_G_scores, dim=0).mean(dim=0)
    hypo_T_scores = torch.stack(hypo_T_scores, dim=0).mean(dim=0)
    importance_scores = torch.stack(importance_scores, dim=0).mean(dim=0)

    # stacking all hypothetical scores

    hypo_scores_tensor = []
    for i in range(seq.shape[0]):
        combined = torch.stack((hypo_A_scores[i, :], hypo_C_scores[i, :], hypo_G_scores[i, :], hypo_T_scores[i, :]),
                               dim=0).unsqueeze(0)
        hypo_scores_tensor.append(combined)

    hypo_scores = torch.cat(hypo_scores_tensor, dim=0)

    # adding one hot sequences in required format
    one_hot_batch = []
    for i in range(seq_ref.shape[0]):
        one_hot.append(seq.detach().cpu().numpy()[i, :, :].T)
        one_hot_batch.append(seq.detach().cpu().numpy()[i, :, :].T)



    # processing importance scores to make them tfmodisco compatible format
    task_to_scores_value = [np.array([imp_array, imp_array, imp_array, imp_array]).T for imp_array in
                            list(importance_scores.detach().cpu().numpy())]
    task_to_scores_value_final = []
    for i in range(len(task_to_scores_value)):
        task_to_scores_value_final.append(np.where(one_hot_batch[i] >= 1, task_to_scores_value[i], 0))

    # appending importance scores and hypothetical importance scores to the dictionary
    task_to_scores['JUND'] = task_to_scores['JUND'] + task_to_scores_value_final

    task_to_hyp_scores['JUND'] = task_to_hyp_scores['JUND'] + [imp_array.T for imp_array in
                                                               list(hypo_scores.detach().cpu().numpy())]

with open("indiv_model_hypo.pkl", "wb") as f:
    pickle.dump(task_to_hyp_scores, f)

with open("indiv_model_imp.pkl", "wb") as f:
    pickle.dump(task_to_scores, f)

with open("indiv_model_onehot_.pkl", "wb") as f:
    pickle.dump(one_hot, f)


