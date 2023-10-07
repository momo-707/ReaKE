import os
import dgl
import torch
import pickle
from molecule import Molecule
from grapher import *
from rdkit import Chem
from dgl.data.utils import save_info, load_info



class YieldPredictionDataset(dgl.data.DGLDataset):
    def __init__(self, args, dname):
        self.args = args
        self.dname = dname
        self.path = '../data/' + args.dataset + '/yield_cache/' + self.dname
        self.reactant_graphs = []
        self.temp_graphs = []
        self.prod_graphs = []
        self.labels = []
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()
        super().__init__(name='yield_prediction_' + dname)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.args.dataset + ' dataset to GPU')
            self.labels = self.labels.to('cuda:' + str(self.args.gpu))
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.prod_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.prod_graphs]

    def save(self):
        print('saving ' + self.args.dataset + ' dataset to ' + self.path + '.bin')
        save_info(self.path + '_info.pkl', {'labels': self.labels})
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.prod_graphs)
        dgl.save_graphs(self.path + '_template_graphs.bin', self.temp_graphs)

    def load(self):
        print('loading ' + self.args.dataset + ' dataset from ' + self.path + '.bin')
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.temp_graphs = dgl.load_graphs(self.path + '_template_graphs.bin')[0]
        self.prod_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]
        self.labels = load_info(self.path + '_info.pkl')['labels']
        #self.to_gpu()

    def process(self):
        print('loading atom_species from saved/' + self.args.pretrained_model + '/atom_species.pkl')
        with open('../saved/' + self.args.pretrained_model + '/atom_species.pkl', 'rb') as f:
            atom_species = pickle.load(f)
        print('processing ' + self.args.dataset + '/' + self.dname + 'dataset')
        original_path = '../data/' + self.args.dataset + '/' + self.dname
        with open(original_path + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == '\n':
                    continue
                items = line.strip().split(',')
                _, reactant, template, prod, label = items[0], items[1], items[2], items[3], items[4]
                reacts, prods, temps = [], [], []
                reactants_smiles = reactant.split('.')
                template_smiles = template.split('.')
                products_smiles = prod.split('.')
                for reactant in reactants_smiles:
                    m = Chem.MolFromSmiles(reactant)
                    m = Molecule(m)
                    reacts.append(m)
                for template in template_smiles:
                    m = Chem.MolFromSmiles(template)
                    m = Molecule(m)
                    temps.append(m)
                for product in products_smiles:
                    m = Chem.MolFromSmiles(product)
                    m = Molecule(m)
                    prods.append(m)
                reactant_graph, product_graph = build_graph_and_featurize_reaction(mode='downstream', reaction=Reaction(reacts, prods, 'smiles'),
                                                                                     atom_featurizer=self.atom_featurizer,
                                                                                     bond_featurizer=self.bond_featurizer,
                                                                                     atom_speices=atom_species)
                template_graph, _ = build_graph_and_featurize_reaction(mode='downstream', reaction=Reaction(temps, prods, 'smiles'),
                                                                                     atom_featurizer=self.atom_featurizer,
                                                                                     bond_featurizer=self.bond_featurizer,
                                                                                     atom_speices=atom_species)
                self.reactant_graphs.append(reactant_graph)
                self.temp_graphs.append(template_graph)
                self.prod_graphs.append(product_graph)
                self.labels.append(float(label))
            self.labels = torch.Tensor(self.labels)

    def has_cache(self):
        return os.path.exists(self.path + '_reactant_graphs.bin') and os.path.exists(self.path + '_product_graphs.bin')

    def __getitem__(self, i):
        return self.reactant_graphs[i], self.temp_graphs[i], self.prod_graphs[i], self.labels[i]

    def __len__(self):
        return len(self.reactant_graphs)

    def get_all_items(self, type):
        if type == 'reactant':
            return self.reactant_graphs
        elif type == 'product':
            return self.prod_graphs
        elif type == 'label':
            return self.labels
        elif type == 'template':
            return self.temp_graphs


def load_data(args):
    NAMES = ["Test1",
        "Test2",
        "Test3",
        "Test4"]

    data = []
    if not os.path.exists('../data/' + args.dataset + '/yield_cache/'):
        path = '../data/' + args.dataset + '/yield_cache/'
        print('creating directory: %s' % path)
        os.mkdir(path)

    for dname in NAMES:
        dataset = YieldPredictionDataset(args, dname)
        data.append(dataset)
    return data
