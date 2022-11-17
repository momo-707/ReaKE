import os
import dgl
import torch
import pickle
from molecule import Molecule
from grapher import *
from rdkit import Chem
from dgl.data.utils import save_info, load_info



class ReactionClassDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.path = '../data/' + args.dataset + '/classification_cache/' + self.mode
        self.reactant_graphs = []
        self.prod_graphs = []
        self.reaction_types = []
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()
        super().__init__(name='reaction_classification_' + mode)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.args.dataset + ' dataset to GPU')
            self.reaction_types = self.reaction_types.to('cuda:' + str(self.args.gpu))
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.prod_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.prod_graphs]

    def save(self):
        print('saving ' + self.args.dataset + ' dataset to ' + self.path + '.bin')
        save_info(self.path + '_info.pkl', {'reaction_types': self.reaction_types})
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.prod_graphs)

    def load(self):
        print('loading ' + self.args.dataset + ' dataset from ' + self.path + '.bin')
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.prod_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]
        self.reaction_types = load_info(self.path + '_info.pkl')['reaction_types']
        self.to_gpu()

    def process(self):
        print('loading atom_species from ../saved/' + self.args.pretrained_model + '/atom_species.pkl')
        with open('../saved/' + self.args.pretrained_model + '/atom_species.pkl', 'rb') as f:
            atom_species = pickle.load(f)
        print('processing ' + self.args.dataset + ' dataset')
        original_path = '../data/' + self.args.dataset + '/' + self.mode
        with open(original_path + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == '\n':
                    continue
                items = line.strip().split(',')
                _, reactant, prod, label = items[0], items[1], items[2], items[3]
                reacts = []
                prods = []
                reactants_smiles = reactant.split('.')
                products_smiles = prod.split('.')
                for reactant in reactants_smiles:
                    if self.args.dataset == 'schneider':
                        m = Chem.MolFromSmiles(reactant)
                    m = Molecule(m)
                    reacts.append(m)
                for product in products_smiles:
                    if self.args.dataset == 'schneider':
                        m = Chem.MolFromSmiles(product)
                    m = Molecule(m)
                    prods.append(m)
                reactant_graph, product_graph = build_graph_and_featurize_reaction(mode='downstream', reaction=Reaction(reacts, prods),
                                                                                     atom_featurizer=self.atom_featurizer,
                                                                                     bond_featurizer=self.bond_featurizer,
                                                                                     atom_speices=atom_species)
                self.reactant_graphs.append(reactant_graph)
                self.prod_graphs.append(product_graph)
                self.reaction_types.append(int(label))
            self.reaction_types = torch.LongTensor(self.reaction_types)

    def has_cache(self):
        return os.path.exists(self.path + '_reactant_graphs.bin') and os.path.exists(self.path + '_product_graphs.bin')

    def __getitem__(self, i):
        return self.reactant_graphs[i], self.prod_graphs[i], self.reaction_types[i]

    def __len__(self):
        return len(self.reactant_graphs)


def load_data(args):
    if os.path.exists('../data/' + args.dataset + '/classification_cache/'):
        train_data = ReactionClassDataset(args, 'train')
        val_data = ReactionClassDataset(args, 'valid')
        test_data = ReactionClassDataset(args, 'test')
    else:
        print('no cache found')
        path = '../data/' + args.dataset + '/classification_cache/'
        print('creating directory: %s' % path)
        os.mkdir(path)
        train_data = ReactionClassDataset(args, 'train')
        val_data = ReactionClassDataset(args, 'valid')
        test_data = ReactionClassDataset(args, 'test')
    return train_data, val_data, test_data
