import os
import argparse
import data_processing
import train
import pandas as pd
from yield_pred import yp_data_preprocess, yp_train
from reaction_class import rc_data_preprocess, rc_train
from visual import rc_data_preprocess_reake, tmap_visual_reake

def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5, help='the index of gpu device')

    #'''
    # pretraining / chemical reaction prediction
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--dataset', type=str, default='MIT', help='dataset name')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--gnn', type=str, default='tag', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of molecule embeddings')
    parser.add_argument('--ratio', type=float, default=0.7, help='drop atoms ratio for subgraph')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model to disk')
    parser.add_argument('--KGmodel', type=str, default='TransE', help='mode for KG')
    #'''
    '''
    parser.add_argument('--task', type=str, default='reaction_classification', help='downstream task')
    parser.add_argument('--visual', type=bool, default=false, help='batch size for calling the pretrained model')
    parser.add_argument('--pretrained_model', type=str, default='tag', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='schneider', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size for calling the pretrained model')
    '''
    '''
    parser.add_argument('--task', type=str, default='yield_prediction', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='tag', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='buchwald_hartwig', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size for calling the pretrained model')
    '''

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    if args.task == 'pretrain':
        data = data_processing.load_data(args)
        train.train(args, data)

    elif args.task == 'reaction_classification':
        #classification
        if args.visual == False:
            data = rc_data_preprocess.load_data(args)
            rc_train.train(args, data)
        #visualization
        else:
            data = rc_data_preprocess_reake.load_data(args)
            tmap_visual_reake.draw(args, data)

    elif args.task == 'yield_prediction':
        data = yp_data_preprocess.load_data(args)
        yp_train.train(args, data)

if __name__ == '__main__':
    main()

