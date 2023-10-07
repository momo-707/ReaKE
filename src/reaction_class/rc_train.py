import torch
import pickle
from dgl.dataloading import GraphDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from dgl.data.utils import load_info
from model import Reaction_KG


def train(args, data):
    #加载模型
    path = '../saved/' + args.pretrained_model + '/'
    path_parameter = path + 'parameter_info.pkl'
    feature_len, type_num = load_info(path_parameter)['feature_length'], load_info(path_parameter)['type_num']

    print('loading hyperparameters of pretrained model from ' + path + 'args.pkl')
    with open(path + 'args.pkl', 'rb') as f:
        hparams = pickle.load(f)

    print('loading pretrained model from ' + path + 'model.pt')
    mole = Reaction_KG(hparams, feature_len, type_num)

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
    with torch.no_grad():
        mole.eval()
        for reactant, prod, labels in train_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = torch.cat((reactant_embeddings, prod_embeddings), 1)  #128*1024
            train_features.append(embeddings)
            train_labels.append(labels)
        train_features = torch.cat(train_features, dim=0).cpu().numpy()
        train_labels = torch.cat(train_labels, dim=0).cpu().numpy()

        for reactant, prod, labels in val_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = torch.cat((reactant_embeddings, prod_embeddings), 1)  # 按维数0拼接（竖着拼）
            val_features.append(embeddings)
            val_labels.append(labels)
        val_features = torch.cat(val_features, dim=0).cpu().numpy()
        val_labels = torch.cat(val_labels, dim=0).cpu().numpy()

        for reactant, prod, labels in test_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = torch.cat((reactant_embeddings, prod_embeddings), 1)  # 按维数0拼接（竖着拼）
            test_features.append(embeddings)
            test_labels.append(labels)
        test_features = torch.cat(test_features, dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    print('training the classification model\n')
    pred_model = LogisticRegression(multi_class='ovr', solver='liblinear')
    pred_model.fit(train_features, train_labels)
    run_classification(pred_model, 'train', train_features, train_labels)
    run_classification(pred_model, 'valid', val_features, val_labels)
    run_classification(pred_model, 'test', test_features, test_labels)


def run_classification(model, mode, features, labels):
    train_acc = model.score(features, labels)
    train_auc = roc_auc_score(labels, model.predict_proba(features), multi_class='ovr')
    print('%s acc: %.4f   auc: %.4f' % (mode, train_acc, train_auc))
