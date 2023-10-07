import torch
import pickle
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import load_info
from model import Reaction_KG
from statistics import stdev
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from math import sqrt
import pandas as pd
from rdkit import Chem
from grapher import *
from molecule import Molecule


def Metric(y_preds,y_labels):
    rmse = sqrt(mean_squared_error(y_labels , y_preds))
    r2 = r2_score(y_labels, y_preds)
    pearson = pearsonr(y_labels, y_preds)[0]
    spearman = spearmanr(y_labels, y_preds)[0]
    return rmse, r2, pearson, spearman


def train(args, data):
    #加载模型
    path = '../saved/' + args.pretrained_model + '/'
    path_parameter = path + 'parameter_info.pkl'
    feature_len = load_info(path_parameter)['feature_length']

    print('loading hyperparameters of pretrained model from ' + path + 'args.pkl')
    with open(path + 'args.pkl', 'rb') as f:
        hparams = pickle.load(f)

    print('loading pretrained model from ' + path + 'model.pt'  )
    mole = Reaction_KG(hparams, feature_len)

    #加载模型
    if torch.cuda.is_available():
        mole.load_state_dict(torch.load(path + 'model.pt'))
        mole = mole.cuda(args.gpu)
    else:
        mole.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))

    #产量预测
    predict_buchwald_hartwig_tests(args, mole, data)

def save_results(model, mode, ratio, avg_r2, std_r2, avg_rmse=0, std_rmse=0, avg_pearson=0, std_pearson=0, avg_spearman=0, std_spearman=0) -> None:
    df = pd.DataFrame({'ratio': ratio,
                        'avg_r2': avg_r2,
                        'std_r2': std_r2,
                        'avg_rmse': avg_rmse,
                        'std_rmse': std_rmse,
                        'avg_pearson': avg_pearson,
                        'std_pearson': std_pearson,
                        'avg_spearman': avg_spearman,
                        'std_spearman': std_spearman})
    df.columns = ['ratio', 'avg_r2', 'std_r2', 'avg_rmse', 'std_rmse', 'avg_pearson', 'std_pearson', 'avg_spearman', 'std_spearman']
    df.to_csv(model+mode+'_result.csv')

def load_data(args, mole, dataset, valid_frac=0.1, split=2767):
    dataloader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    all_features = []
    all_labels = []

    with torch.no_grad():
        mole.eval()
        for reactant, temp, prod, labels in dataloader:
            if torch.cuda.is_available():
                reactant, temp, prod = reactant.to('cuda:' + str(args.gpu)), temp.to('cuda:' + str(args.gpu)), prod.to('cuda:' + str(args.gpu))
            reactant_embeddings = mole(h=reactant, mode='downstream')
            type_embeddings = mole(h=temp, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = torch.cat((reactant_embeddings - prod_embeddings, type_embeddings), 1)
            all_features.append(embeddings)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()


        X_test = all_features[split:]
        X_train = all_features[:split]

        y_test = all_labels[split:]
        y_train = all_labels[:split]
        np.random.seed(42)
        valid_indices = np.random.choice(
            np.arange(len(X_train)), int(valid_frac * len(X_train)), replace=False
        )

        X_valid = X_train[valid_indices]
        y_valid = y_train[valid_indices]

        train_indices = list(set(range(len(X_train))) - set(valid_indices))

        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        return X_train, y_train, X_valid, y_valid, X_test, y_test

def predict_buchwald_hartwig_tests(args, mole, data):

    buchwald_hartwig_yield_fps, labels = [], []
    ratio, avg_r2, std_r2 = [], [], []
    avg_rmse, std_rmse = [], []
    avg_pearson, std_pearson = [], []
    avg_spearman, std_spearman = [], []

    for i in range(4):
        buchwald_hartwig_yield_fps.append(data[i])

    r2s_all, rmse_all = [], []
    for i, sample_file in enumerate(buchwald_hartwig_yield_fps):
        rmses, r2s, pearsons, spearmans = [], [], [], []
        for seed in [42, 69, 2222, 2626]:
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(args, mole, sample_file, split=3058)

            model = XGBRegressor(
                nthread=15,
                n_estimators=999999,
                learning_rate=0.01,
                max_depth=12,
                min_child_weight=6,
                colsample_bytree=0.6,
                subsample=0.8,
                random_state=seed,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=20,
                verbose=False,
            )

            y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
            y_pred[y_pred < 0.0] = 0.0

            rmse, r2, pearson, spearman = Metric(y_pred, y_test)
            rmses.append(rmse)
            r2s.append(r2)
            pearsons.append(pearson)
            spearmans.append(spearman)

        r2s_all.append(r2s)
        rmse_all.append(rmses)
        ratio.append(i+1)
        avg_r2.append(sum(r2s) / len(r2s))
        std_r2.append(stdev(r2s))
        avg_rmse.append(sum(rmses) / len(rmses))
        std_rmse.append(stdev(rmses))
        avg_pearson.append(sum(pearsons) / len(pearsons))
        std_pearson.append(stdev(pearsons))
        avg_spearman.append(sum(spearmans) / len(spearmans))
        std_spearman.append(stdev(spearmans))
        print(f"Test {i + 1}")
        print("avg_r2: %.4f, std_r2:  %.4f"%(sum(r2s) / len(r2s), stdev(r2s)))
        print("avg_rmse: %.4f, std_rmse:  %.4f"%(sum(rmses) / len(rmses), stdev(rmses)))
        print("avg_pearson: %.4f, std_pearson:  %.4f"%(sum(pearsons) / len(pearsons), stdev(pearsons)))
        print("avg_spearman: %.4f, std_spearman:  %.4f"%(sum(spearmans) / len(spearmans), stdev(spearmans)))
    r2s_all = np.array(r2s_all).reshape(-1)
    rmse_all = np.array(rmse_all).reshape(-1)
    save_results(args.pretrained_model, 'test', ratio, avg_r2, std_r2, avg_rmse, std_rmse, avg_pearson, std_pearson, avg_spearman, std_spearman)
    print("Test1-Test4 avg_r2s: %.4f, std_r2s: %.4f"%(sum(r2s_all) / len(r2s_all), stdev(r2s_all)))
    print("Test1-Test4 avg_rmses: %.4f, std_rmses: %.4f"%(sum(rmse_all) / len(rmse_all), stdev(rmse_all)))
    df = pd.DataFrame({'r2s_all': r2s_all, 'rmse_all': rmse_all})
    df.to_csv('all_test_result.csv')