import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F

class teacher_model(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(teacher_model, self).__init__()
        self.first_linear_dim = dim_in
        self.ffn_hidden_size = 100
        self.output_size = dim_out
        self.sigmoid = nn.Sigmoid()

        self.first = nn.Linear(self.first_linear_dim, self.ffn_hidden_size)
        self.hidden1 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden2 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden3 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.out = nn.Linear(self.ffn_hidden_size, self.output_size)

        initialize_weights(self)

    def forward(self, X):
        output = F.relu(self.first(X))
        output = F.relu(self.hidden1(output))
        output = F.relu(self.hidden2(output))
        output = F.relu(self.hidden3(output))
        output = self.out(output)
        output = self.sigmoid(output)

        return output
with open('Data/teacher_model_positive_gene.txt', 'r')as f:
    dict_positive_gene = {}
    readlines = f.readlines()
    for line in readlines:
        gene = line.strip('\n')
        dict_positive_gene[gene] = 'yes'
with open('Data/Ppi_feature.txt', 'r')as f:
    dict_PPI = {}
    readlines = f.readlines()
    for line in readlines:
        gene = line.strip('\n').split('\t')[0]
        shuzhi = line.strip('\n').split('\t')[1]
        dict_PPI[gene] = float(shuzhi)
def initialize_weights(model: nn.Module):
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
def load_checkpoint(path: str):
    debug = info = print
    state = torch.load(path)
    loaded_state_dict = state['state_dict']
    model = teacher_model(50,1)
    model_state_dict = model.state_dict()
    pretrained_state_dict = {}

    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            info(f'Warning: Pretrained parameter "{param_name}" '
                 f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model
def prepare_feature():
    with open('Data/Omic_feature.txt','r')as f:
        one_feature = []
        name = []
        readlines = f.readlines()
        for line in readlines:
            uniprotid = line.strip('\n').split('\t')[0]
            feature1 = [float(i) for i in line.strip('\n').split('\t')[1:]]
            if uniprotid in dict_PPI:
                feature1.append(dict_PPI[uniprotid])
            else:
                feature1.append(0.0)
            one_feature.append(feature1)
            name.append(uniprotid)


    one_feature = torch.Tensor(one_feature)
    X_final = one_feature

    label = []
    for i in range(len(name)):
        if name[i] in dict_positive_gene:
            label.append([1])
        else:
            label.append([0])


    X_label = torch.Tensor(label)

    return X_final,X_label,name
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='orange', label='AUC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    #plt.savefig('teacher_model_auc.png')
    plt.close()

if __name__ == '__main__':
    model1 = load_checkpoint('Model/teacher_model')

    X_final,X_label,name = prepare_feature()
    y_pred = model1(X_final)

    auc = roc_auc_score(X_label.detach().numpy(), y_pred.detach().numpy())
    fper, tper, thresholds = roc_curve(X_label.detach().numpy(), y_pred.detach().numpy())
    plot_roc_curve(fper, tper)
    print('teacher_model_auc:', auc)

    pre_label = X_label.tolist()
    pre_value = y_pred.tolist()
    with open('Data/teacher_model_predictor_result.txt', 'w')as f:
        for i in range(len(pre_label)):
            f.write(name[i]+'\t'+str(pre_value[i][0]) + '\t' + str(pre_label[i][0]) + '\n')