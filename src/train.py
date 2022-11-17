import os
import torch
import pickle
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Reaction_KG, KGLoss
from copy import deepcopy
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info
from torch.nn.functional import one_hot, normalize, logsigmoid


def train(args, data):
    argsDict = args.__dict__
    train_data, valid_data, test_data = data

    path_parameter = '../data/' + args.dataset + '/cache/train_parameter_info.pkl'
    path_datainfo = '../data/' + args.dataset + '/cache/train_data_info.pkl'
    path_atom_species = '../data/' + args.dataset + '/cache/atom_species.pkl'
    feature_len, num_samples, num_triplets = load_info(path_parameter)['feature_length'], load_info(path_parameter)['num_samples'], load_info(path_parameter)['num_triplets']
    num_templates, num_reaction_bf = load_info(path_datainfo)['num_templates'], load_info(path_datainfo)['num_reaction_bf']
    with open(path_atom_species, 'rb') as f:
        atom_species = pickle.load(f)
    print("原子的特征长度为: %d" % (feature_len))
    print("总共有%d个反应，在去除无法转换成图的模版后有%d个反应" % (num_reaction_bf, num_samples))
    print("总共有%d个三元组，总共有%d种templates" % (num_triplets, num_templates))

    model = Reaction_KG(argsDict, feature_len)
    calculate_loss = KGLoss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                              patience=5, verbose=True, threshold=0.001, threshold_mode='rel',
                                              cooldown=10,
                                              min_lr=0, eps=1e-08)

    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_dataloader = GraphDataLoader(valid_data, batch_size=args.batch_size)
    test_dataloader = GraphDataLoader(test_data, batch_size=args.batch_size)

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)

    best_model_params = None
    best_val_mrr = 0
    best_val_mr = 100
    print('start training\n')

    print('initial case:')
    model.eval()

    evaluate(model, 'valid', valid_dataloader, args)
    evaluate(model, 'test', test_dataloader, args)

    print()

    for i in range(args.epoch):
        print('epoch %d:' % i)

        j=0
        # train
        model.train()
        all_rankings = []
        for reactant_graphs, template_r, template_p, product_graphs in train_dataloader:
            if torch.cuda.is_available():
                reactant_graphs, template_r, template_p, product_graphs= reactant_graphs.to('cuda:' + str(args.gpu)) , template_r.to('cuda:' + str(args.gpu)), template_p.to('cuda:' + str(args.gpu)) , product_graphs.to('cuda:' + str(args.gpu))
            #print("1:{}".format(torch.cuda.memory_allocated()))
            reactant_embeddings, reaction_type_embedding, product_embeddings, aug_loss = model(reactant_graphs, template_r, template_p, product_graphs)
            loss = calculate_loss(reactant_embeddings, reaction_type_embedding, product_embeddings, aug_loss, 'train')
            if j==0: print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j == 0:
                ground_truth = torch.unsqueeze(torch.arange(0, args.batch_size), dim=1)            
                if torch.cuda.is_available():
                    ground_truth = ground_truth.cuda(args.gpu)
                dist = calculate_loss(reactant_embeddings, reaction_type_embedding, product_embeddings, 0, 'eval')
                sorted_indices = torch.argsort(dist, dim=1)
                rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
                all_rankings = np.array(rankings)
                mrr = float(np.mean(1 / all_rankings))
                mr = float(np.mean(all_rankings))
                h1 = float(np.mean(all_rankings <= 1))
                h3 = float(np.mean(all_rankings <= 3))
                h5 = float(np.mean(all_rankings <= 5))
                h10 = float(np.mean(all_rankings <= 10))

                print('%s  mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % ('train', mrr, mr, h1, h3, h5, h10))

            j += args.batch_size
            del reactant_graphs, template_r, template_p, product_graphs, reactant_embeddings, reaction_type_embedding, product_embeddings, aug_loss, loss
            #torch.cuda.empty_cache()
            #print("5:{}".format(torch.cuda.memory_allocated()))


        # evaluate on the validation set
        val_mrr = evaluate(model, 'valid', valid_dataloader, args)
        evaluate(model, 'test', test_dataloader, args)
        scheduler.step(val_mrr)

        # save the best model
         
         
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())


        print()
    if args.epoch == 0:
        best_model_params = deepcopy(model.state_dict())
    # evaluation on the test set
    print('final results on the test set:')
    model.load_state_dict(best_model_params)
    evaluate(model, 'test', test_dataloader, args)
    print()

    # save the model, hyperparameters, and feature encoder to disk
    if args.save_model:
        if not os.path.exists('../saved/'):
            print('creating directory: ../saved/')
            os.mkdir('../saved/')

        directory = '../saved/%s_%.5f_%s_%.1f_%.1f' % (args.gnn, args.lr, args.KGmodel, args.margin, args.ratio)
        if not os.path.exists(directory):
            os.mkdir(directory)

        print('saving the model to directory: %s' % directory)
        torch.save(best_model_params, directory + '/model.pt')
        with open(directory + '/args.pkl', 'wb') as f:
            pickle.dump(argsDict, f)
        with open(directory + '/atom_species.pkl', 'wb') as f:
            pickle.dump(atom_species, f)

        save_info(directory + '/parameter_info.pkl', {'feature_length': feature_len})


def evaluate(model, mode, dataloader, args):
    calculate_loss = KGLoss(args)
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        all_product_embeddings = []
        for _, _, template_ps, product_graphs in dataloader:
            if torch.cuda.is_available():
                template_ps, product_graphs = template_ps.to('cuda:'+str(args.gpu)), product_graphs.to('cuda:' + str(args.gpu))
            product_embeddings = model(h=product_graphs, r_r=template_ps, mode='test')
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)
        data_len = all_product_embeddings.shape[0]
        # rank
        all_rankings = []
        i = 0
        for reactant_graphs, template_rs, _, _ in dataloader:
            if torch.cuda.is_available():
                reactant_graphs, template_rs = reactant_graphs.to('cuda:' + str(args.gpu)), template_rs.to('cuda:' + str(args.gpu))
            reactant_embeddings = model(h=reactant_graphs, r_r=template_rs, mode='test')
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch_size, data_len)), dim=1)
            i += args.batch_size
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda(args.gpu)
            dist = calculate_loss(head=reactant_embeddings, tail=all_product_embeddings, mode='eval')
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)

        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))

        del reactant_graphs, template_rs, template_ps, product_graphs, reactant_embeddings, product_embeddings
        torch.cuda.empty_cache()

        print('%s  mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % (mode, mrr, mr, h1, h3, h5, h10))
        return h10