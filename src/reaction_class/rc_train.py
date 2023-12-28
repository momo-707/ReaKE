import torch
import pickle
from model import GNN
from dgl.dataloading import GraphDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from dgl.data.utils import load_info
from model import Reaction_KG
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
from statistics import stdev

def train(args, data):
    #加载模型
    path = '../saved/' + args.pretrained_model + '/'
    path_parameter = path + 'parameter_info.pkl'
    feature_len = load_info(path_parameter)['feature_length']

    print('loading hyperparameters of pretrained model from ' + path + 'args.pkl')
    with open(path + 'args.pkl', 'rb') as f:
        hparams = pickle.load(f)

    print('loading pretrained model from ' + path + 'model.pt')
    mole = Reaction_KG(hparams, feature_len)

    if torch.cuda.is_available():
        mole.load_state_dict(torch.load(path + 'model.pt'))
        mole = mole.cuda(args.gpu)
    else:
        mole.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))


    #数据load
    train_data, val_data, test_data = data
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_dataloader = GraphDataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = GraphDataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    test_features = []
    test_labels = []
    type_index = []
    with torch.no_grad():
        mole.eval()
        for reactant, prod, labels in train_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = reactant_embeddings - prod_embeddings  # 128*1024
            train_features.append(embeddings)
            train_labels.append(labels)
        train_features = torch.cat(train_features, dim=0).cpu().numpy()
        train_labels = torch.cat(train_labels, dim=0).cpu().numpy()
        type_num_max = max(train_labels)
        type_num_min = min(train_labels)
        for i in range(type_num_min, type_num_max + 1):
            index_list = np.argwhere(train_labels == i)
            index_list = index_list[:, 0]
            print("反应类型%d：%d个反应" % (i, len(index_list)))
            type_index.append(index_list)
        print(len(type_index))


        for reactant, prod, labels in val_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = reactant_embeddings - prod_embeddings
            val_features.append(embeddings)
            val_labels.append(labels)
        val_features = torch.cat(val_features, dim=0).cpu().numpy()
        val_labels = torch.cat(val_labels, dim=0).cpu().numpy()

        for reactant, prod, labels in test_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = reactant_embeddings - prod_embeddings
            test_features.append(embeddings)
            test_labels.append(labels)
        test_features = torch.cat(test_features, dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    times=5
    per_class = [4, 8, 16, 32, 64, 128] #schneider
    #per_class = [4]  # test for ablation study
    #per_class = [4, 8, 16, 32, 64] #grambow
    for i in per_class:
        f1s = []
        for n in range(times):
            train_index = []
            for k, j in enumerate(type_index):
                np.random.seed(k+n)
                class_index = np.random.choice(j, i, replace=False)
                train_index.append(class_index)
            train_index = np.array(train_index).reshape(-1)
            mini_train_features = train_features[train_index]
            mini_train_labels = train_labels[train_index]
            #'''
            pred_model = MLPClassifier(
                hidden_layer_sizes=(400, 200),
                activation='relu',
                solver='adam', max_iter=200, random_state=1)
            #'''
            #pred_model = LogisticRegression(multi_class='ovr', solver='liblinear')
            pred_model.fit(mini_train_features, mini_train_labels)
            print("%d reactions per class as train data" % (i))
            print('training the classification model\n')
            run_classification(pred_model, 'train', mini_train_features, mini_train_labels)
            run_classification(pred_model, 'valid', val_features, val_labels)
            f1 = run_classification(pred_model, 'test', test_features, test_labels)
            f1s.append(f1)
        avg = sum(f1s) / len(f1s)
        std = stdev(f1s)
        print('%d reactions per class, avg f1_score: %.4f, std: %.4f' % (i, avg, std))


def run_classification(model, mode, features, labels):
    train_acc = model.score(features, labels)
    train_auc = roc_auc_score(labels, model.predict_proba(features), multi_class='ovr')
    pred = model.predict(features)
    f1 = f1_score(labels, pred, average='micro')
    print('%s acc: %.4f   auc: %.4f    f1_score：%.4f' % (mode, train_acc, train_auc, f1))
    return f1
