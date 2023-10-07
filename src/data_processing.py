import os
from triplet import Triplet
from rdkit import Chem
import itertools
from dgl.data.utils import save_info, load_info
from grapher import *
import pickle


class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode, reactions=None, atom_speices=None, templates=None):
        self.args = args
        self.mode = mode
        self.reactions = reactions
        self.templates = templates
        self.reactant_graphs = []
        self.product_graphs = []
        self.template_rs = []
        self.template_ps = []
        self.num_samples = 0
        self.num_triplets = 0
        self.atom_species = atom_speices
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()
        self.path = '../data/' + self.args.dataset + '/cache/' + self.mode
        self.feature_length = int(0)
        self.type_num = int(0)
        super().__init__(name='Smiles_' + mode)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.mode + ' data to GPU')
            self.reaction_types = self.reaction_types.to('cuda:' + str(self.args.gpu))
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.product_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.product_graphs]
            self.template_rs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.template_rs]
            self.template_ps = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.template_ps]

    def save(self):
        print('saving ' + self.mode + ' reactants graph to ' + self.path + '_reactant_graphs.bin')
        print('saving ' + self.mode + ' products graph to ' + self.path + '_product_graphs.bin')
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.product_graphs)
        dgl.save_graphs(self.path + '_template_rs.bin', self.template_rs)
        dgl.save_graphs(self.path + '_template_ps.bin', self.template_ps)
        save_info(self.path + '_parameter_info.pkl', {'feature_length': self.feature_length, 'num_samples':self.num_samples, 'num_triplets':self.num_triplets})

    def load(self):
        print('loading ' + self.mode + ' reactants from ' + self.path + '_reactant_graphs.bin')
        print('loading ' + self.mode + ' products from ' + self.path + '_product_graphs.bin')
        # graphs loaded from disk will have a default empty label set: [graphs, labels], so we only take the first item
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.product_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]
        self.template_rs = dgl.load_graphs(self.path + '_template_rs.bin')[0]
        self.template_ps = dgl.load_graphs(self.path + '_template_ps.bin')[0]
        #self.to_gpu()

    def process(self):
        print('transforming ' + self.mode + ' data to DGL graphs')
        temp_rs, temp_ps = [], []
        for id, (reaction, template) in enumerate(zip(self.reactions, self.templates)):
            if id % 10000 == 0:
                print('%dw' % (id // 10000))
            if not template.mol_check():
                print('No.%d reaction do not have valid template'%id)
                continue
            #生成模板图
            template_rg, template_pg= build_graph_and_featurize_reaction(mode='template', reaction=template,atom_featurizer=self.atom_featurizer,bond_featurizer=self.bond_featurizer,
                                                                                atom_speices=self.atom_species)
            temp_rs.append(template_rg)
            temp_ps.append(template_pg)
            #生成反应图
            reactant_graphs, product_graphs= build_graph_and_featurize_reaction(mode=self.mode, reaction=reaction,atom_featurizer=self.atom_featurizer,bond_featurizer=self.bond_featurizer,
                                                                                atom_speices=self.atom_species)
            if self.mode == "train":
                KG_triplet = Triplet(reaction = reaction, functional_group_smarts_filenames='smarts_daylight.tsv')
                #为分子graph添加官能团节点属性
                reactant_graphs_final, product_graphs_final = KG_triplet(reactant_graphs, product_graphs)
                for i in itertools.product(reactant_graphs_final, product_graphs_final):#返回h,r,t所有可能的组合
                    self.reactant_graphs.append(i[0])
                    self.template_rs.append(temp_rs[self.num_samples])
                    self.template_ps.append(temp_ps[self.num_samples])
                    self.product_graphs.append(i[1])
            else:
                self.reactant_graphs.append(reactant_graphs)
                self.product_graphs.append(product_graphs)
                self.template_rs.append(temp_rs[self.num_samples])
                self.template_ps.append(temp_ps[self.num_samples])
            self.num_samples += 1
        self.feature_length = self.reactant_graphs[0].ndata['atom'].shape[1]
        self.num_triplets = len(self.reactant_graphs)
        print("去除不合法的template之后总共有%d个反应" % self.num_samples)
        #self.to_gpu()

    def has_cache(self):
        return os.path.exists(self.path + '_reactant_graphs.bin') and os.path.exists(self.path + '_product_graphs.bin')

    def __getitem__(self, i):
        return self.reactant_graphs[i], self.template_rs[i], self.template_ps[i], self.product_graphs[i]

    def __len__(self):
        return len(self.reactant_graphs)


def read_data(dataset, mode):
    path = '../data/' + dataset + '/' + mode + '.csv'
    print('preprocessing %s data from %s' % (mode, path))

    reactions= []
    templates = []
    temp = []
    species = set()

    with open(path) as f:
        for line in f.readlines():
            id, reactant_smiles, product_smiles, template_r, template_p = line.strip().split(',')
            # skip the first line
            if len(id) == 0:
                continue
            temp.append(template_r + '<<' + template_p)
            if int(id) % 10000 == 0:
                print('%dw' % (int(id) // 10000))

            if '[se]' in reactant_smiles:
                reactant_smiles = reactant_smiles.replace('[se]', '[Se]')
            if '[se]' in product_smiles:
                product_smiles = product_smiles.replace('[se]', '[Se]')

            reacts, prods, temp_rs, temp_ps = [], [], [], []
            reactants_smiles, products_smiles = reactant_smiles.split('.'), product_smiles.split('.')
            template_r, template_p = template_r.split('.'), template_p.split('.')
            for reactant in reactants_smiles:
                m = Chem.MolFromSmiles(reactant)
                m = Molecule(m)
                if mode == 'train':
                    species.update(m.species)
                reacts.append(m)
            for product in products_smiles:
                m = Chem.MolFromSmiles(product)
                m = Molecule(m)
                if mode == 'train':
                    species.update(m.species)
                prods.append(m)
            for tr in template_r:
                m = Chem.MolFromSmarts(tr)
                m.UpdatePropertyCache(strict=False)
                m = Molecule(m)
                temp_rs.append(m)
            for tp in template_p:
                m = Chem.MolFromSmarts(tp)
                m.UpdatePropertyCache(strict=False)
                m = Molecule(m)
                temp_ps.append(m)
            reactions.append(Reaction(reacts, prods))
            templates.append(Reaction(temp_rs, temp_ps, type='smiles', sanity_check = False))
    num_templates = len(list(set(temp)))
    print("总共有%d种反应模版" % num_templates)
    num_reactions = len(reactions)
    print("总共有%d个反应" % num_reactions)
    path = '../data/' + dataset + '/cache/' + mode
    save_info(path + '_data_info.pkl', {'num_templates': num_templates, 'num_reaction_bf': num_reactions})
    if mode == 'train':
        atom_species = sorted(species)
        return reactions, atom_species, templates
    else:
        return reactions, templates


def preprocess(dataset):
    print('preprocessing %s dataset' % dataset)

    # read all data and get all values for attributes
    train_reactions, atom_species, train_types = read_data(dataset, 'train')
    valid_reactions, valid_types = read_data(dataset, 'valid')
    test_reactions, test_types = read_data(dataset, 'test')
    path = '../data/' + dataset + '/cache/atom_species.pkl'
    print('saving atom_species to %s' % path)
    with open(path, 'wb') as f:
        pickle.dump(atom_species, f)
    return atom_species, train_reactions, valid_reactions, test_reactions, train_types, valid_types, test_types

def load_data(args):
    # if datasets are already cached, skip preprocessing
    if os.path.exists('../data/' + args.dataset + '/cache/'):
        path = '../data/' + args.dataset + '/cache/atom_species.pkl'
        print('cache found\nloading atom_species from %s' % path)
        with open(path, 'rb') as f:
            atom_species = pickle.load(f)
        train_dataset = SmilesDataset(args, 'train')
        valid_dataset = SmilesDataset(args, 'valid')
        test_dataset = SmilesDataset(args, 'test')

    else:
        print('no cache found')
        path = '../data/' + args.dataset + '/cache/'
        print('creating directory: %s' % path)
        os.mkdir(path)
        atom_species, train_reactions, valid_reactions, test_reactions, train_types, valid_types, test_types = preprocess(args.dataset)
        train_dataset = SmilesDataset(args, 'train', train_reactions, atom_species, train_types)
        valid_dataset = SmilesDataset(args, 'valid', valid_reactions, atom_species, valid_types)
        test_dataset = SmilesDataset(args, 'test', test_reactions, atom_species, test_types)

    return train_dataset, valid_dataset, test_dataset
