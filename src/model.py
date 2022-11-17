import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from GNN_layer import GNN


class Reaction_KG(torch.nn.Module):
    def __init__(self, args, feature_len):
        super(Reaction_KG, self).__init__()
        torch.manual_seed(1234)
        torch.set_printoptions(profile="full")
        self.args = args
        self.gnn = args['gnn']
        self.n_layer = args['layer']
        self.dim = args['dim']
        self.KGmodel = args['KGmodel']
        self.ratio = args['ratio']
        self.feature_len = feature_len
        self.GNN_mol = GNN(self.gnn, self.n_layer, self.feature_len, self.dim)
        if self.KGmodel == 'RotatE':
            self.GNN_rel = GNN(self.gnn, self.n_layer, self.feature_len, self.dim // 4)
        else:
            self.GNN_rel = GNN(self.gnn, self.n_layer, self.feature_len, self.dim)


    def Subgraph_ori(self, graph):
        center_atoms = graph.ndata['is_functional_group']
        in_center = torch.where(center_atoms > 0)
        in_center = in_center[0].cpu().numpy()
        out_center = list(set(range(graph.num_nodes())) - set(in_center))
        num_in_center = len(in_center)
        num_out_center = len(out_center)
        num_sample_to_drop = int(self.ratio * num_out_center)
        num_sample = num_out_center - num_sample_to_drop

        sub_graph = list(in_center)
        neigh = np.concatenate([graph.successors(n).cpu().numpy() for n in sub_graph])
        neigh = set(neigh).union(sub_graph).difference(sub_graph)

        while len(sub_graph) < num_in_center + num_sample:

            if len(neigh) == 0:  # e.g. H--H --> H + H, or all atoms included
                break

            sample_atom = np.random.choice(list(neigh))

            assert (
                    sample_atom not in sub_graph
            ), "Something went wrong, this should not happen"

            sub_graph.append(sample_atom)
            neigh = neigh.union(graph.successors(sample_atom).cpu().numpy())

            # remove subgraph atoms from neigh
            neigh = neigh.difference(sub_graph)

        # extract subgraph
        selected = sorted(sub_graph)
        if torch.cuda.is_available():
            result_sub_graph = dgl.node_subgraph(graph, selected).to('cuda:' + str(self.args['gpu']))
        else:
            result_sub_graph = dgl.node_subgraph(graph, selected)
        
        del sub_graph, neigh, sample_atom
        torch.cuda.empty_cache()

        return result_sub_graph


    def Subgraph(self, graph):
        center_atoms = graph.ndata['is_functional_group']
        num_atoms = graph.num_nodes()
        out_center = torch.where(center_atoms == 0)
        out_center  = out_center[0].cpu().numpy()
        drop_number = int(len(out_center) * self.ratio)
        to_drop = np.random.choice(out_center, drop_number, replace=False)
        to_keep = sorted(set(range(num_atoms)) - set(to_drop))
        result_sub_graph = dgl.node_subgraph(graph, to_keep)
        if torch.cuda.is_available():
            result_sub_graph = result_sub_graph.to('cuda:' + str(self.args['gpu']))

        del center_atoms, out_center, to_drop, to_keep
        torch.cuda.empty_cache()

        return result_sub_graph


    def forward(self, h=None, r_r=None, r_p=None, t=None, mode='train'):
        if mode == 'train':
            #h编码
            h_original = self.GNN_mol(h)
            h_mol_aug1 = self.Subgraph(h)
            h_aug1 = self.GNN_mol(h_mol_aug1)
            h_mol_aug2 = self.Subgraph(h)
            h_aug2 = self.GNN_mol(h_mol_aug2)

            #r编码
            relation_r = self.GNN_rel(r_r)
            relation_p = self.GNN_rel(r_p)
            relation_embedding = relation_r - relation_p
            if self.KGmodel == 'RotatE':
                pi = 3.14159265358979323846
                uniform_range = 6 / np.sqrt(self.dim)
                relation_embedding = relation_embedding / (uniform_range / pi)


            #t编码
            t_original = self.GNN_mol(t)
            t_mol_aug1 = self.Subgraph(t)
            t_aug1 = self.GNN_mol(t_mol_aug1)
            t_mol_aug2 = self.Subgraph(t)
            t_aug2 = self.GNN_mol(t_mol_aug2)

            aug_loss = torch.dist(h_aug1, h_aug2) - torch.dist(t_aug1, t_aug2)
            if aug_loss < 0: aug_loss = -aug_loss


            return h_original, relation_embedding, t_original, aug_loss
        elif mode == 'test_hr':
            #h编码
            h_original = self.GNN_mol(h)
            #r编码
            relation_r = self.GNN_rel(r_r)
            relation_p = self.GNN_rel(r_p)
            relation_embedding = relation_r - relation_p
            if self.KGmodel == 'RotatE':
                pi = 3.14159265358979323846
                uniform_range = 6 / np.sqrt(self.dim)
                relation_embedding = relation_embedding / (uniform_range / pi)
            return h_original, relation_embedding
        elif mode == 'test':
            relation = self.GNN_rel(r_r)
            h = self.GNN_mol(h)
            return relation+h
        else:
            return self.GNN_mol(h)



class KGLoss(torch.nn.Module):
    def __init__(self, args):
        super(KGLoss, self).__init__()
        self.KGmodel = args.KGmodel
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.margin = args.margin
        self.alpha = args.alpha

    def forward(self, head, relation=None, tail=None, aug_loss=None, mode='train'):
        if self.KGmodel == 'TransE' and mode == 'train':
            dist = torch.cdist(head+relation, tail, p=2)
        elif self.KGmodel == 'RotatE':
            re_head, im_head = torch.chunk(head, 2, dim=1)
            re_tail, im_tail = torch.chunk(tail, 2, dim=1)
            re_relation = torch.cos(relation)
            im_relation = torch.sin(relation)
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            h = torch.stack([re_score, im_score], dim=0)
            t = torch.stack([re_tail, im_tail], dim=0)
            dist = torch.cdist(h, t, p=2)
            dist = dist.norm(dim=0)
        else:
            dist = torch.cdist(head, tail, p=2)
        if mode == 'train':
            #negative sample
            negative_pairs = torch.cdist(relation, relation, p=1)
            negative_pairs = torch.sign(negative_pairs)
            if torch.cuda.is_available():
                negative_pairs = negative_pairs.cuda(self.gpu)
            negative_nums_h = torch.sum(negative_pairs, dim=1) #固定h和r，对t进行负采样
            negative_nums_t = torch.sum(negative_pairs, dim=0) #固定r和t，对h进行负采样
            pos = torch.diag(dist)  # pos为对角线元素
            neg = negative_pairs * dist + (1-negative_pairs) * self.margin
            neg = torch.relu(self.margin - neg)
            sample_loss_h = torch.sum(neg, dim=1)/negative_nums_h 
            sample_loss_t = torch.sum(neg, dim=0)/negative_nums_t
            sample_loss = sample_loss_h + sample_loss_t
            loss = torch.mean(pos) + (torch.sum(sample_loss) + aug_loss)/self.batch_size
        else: loss = dist


        return loss
