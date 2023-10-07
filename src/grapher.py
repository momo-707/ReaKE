from typing import List, Optional, Tuple, Union
from collections import defaultdict

import dgl
import torch
import itertools

from molecule import Molecule
from reaction import Reaction
from featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer



"""
Build dgl graphs from molecules.

Atoms -> nodes, bonds -> edges (two edges per bond), and optionally a global node for
molecular level features.
"""


def mol_to_graph(mol: Molecule, num_global_nodes: int = 0) -> dgl.DGLGraph:
    edges_dict = defaultdict(list)
    # global nodes
    if num_global_nodes > 0:
        a2a = []
        for i, j in mol.bonds:
            a2a.extend([[i, j], [j, i]])

        edges_dict[("atom", "bond", "atom")] = a2a
        num_nodes_dict = {"atom": mol.num_atoms}
        a2v = []
        v2a = []
        for a in range(mol.num_atoms):
            for v in range(num_global_nodes):
                a2v.append([a, v])
                v2a.append([v, a])

        edges_dict[("atom", "a2g", "global")] = a2v
        edges_dict[("global", "g2a", "atom")] = v2a
        num_nodes_dict["global"] = num_global_nodes
        g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)
    else:
        # atom to atom nodes
        atom_start = []
        atom_end = []
        for i, j in mol.bonds:
            atom_start.append(i)
            atom_end.append(j)
        g = dgl.graph((atom_start, atom_end), num_nodes=mol.num_atoms)
        g = dgl.to_bidirected(g, copy_ndata=True)
    return g


def combine_graphs(
    graphs: List[dgl.DGLGraph],
) -> dgl.DGLGraph:

    num_nodes = 0
    edges_dict = defaultdict(list)

    # reorder atom nodes
    for i, g in enumerate(graphs):
        u, v, eid = g.edges(form="all", order="eid")
        src = [j+num_nodes for j in u]
        dst = [j+num_nodes for j in v]
        edges_dict["atom"].extend([(s, d) for s, d in zip(src, dst)])
        num_nodes += g.number_of_nodes()
    edges = edges_dict["atom"]

    # create graph
    new_g = dgl.graph(edges, num_nodes=num_nodes)

    for ntype in graphs[0].ntypes:
        feat_dicts = [g.nodes[ntype].data for g in graphs]

        # concatenate features
        keys = feat_dicts[0].keys()
        new_feats = {k: torch.cat([fd[k] for fd in feat_dicts], 0) for k in keys}

        new_g.ndata["atom"] = new_feats["atom"]

    return new_g


def create_reaction_graph(
    reactants_graph: dgl.DGLGraph,
    products_graph: dgl.DGLGraph,
    num_unchanged_bonds: int,
    num_lost_bonds: int,
    num_added_bonds: int,
    num_global_nodes: int = 0,
) -> dgl.DGLGraph:
    """
    Create a reaction graph from the reactants graph and the products graph.

    It is expected to take the difference of features between the products and the
    reactants.

    The created graph has the below characteristics:
    1. has the same number of atom nodes as in the reactants and products;
    2. the bonds (i.e. atom to atom edges) are the union of the bonds in the reactants
       and the products. i.e. unchanged bonds, lost bonds in reactants, and added bonds
       in products;
    3. we create `num_global_nodes` global nodes and each id bi-direct connected to
       every atom node.

    This assumes the lost bonds in the reactants (or added bonds in the products) have
    larger node number than unchanged bonds. This is the case if
    :meth:`Reaction.get_reactants_bond_map_number()`
    and
    :meth:`Reaction.get_products_bond_map_number()`
    are used to generate the bond map number when `combine_graphs()`.

    This also assumes the reactants_graph and products_graph are created by
    `combine_graphs()`.

    The order of the atom nodes is unchanged. The bonds
    (i.e. `bond` edges between atoms) are reordered. Actually, The bonds in the
    reactants are intact and the added bonds in the products are updated to come after
    the bonds in the reactants.
    More specifically, according to :meth:`Reaction.get_reactants_bond_map_number()`
    and :meth:`Reaction.get_products_bond_map_number()`, bond 0, 1, ..., N_un-1 are the
    unchanged bonds, N_un, ..., N-1 are the lost bonds in the reactants, and N, ...,
    N+N_add-1 are the added bonds in the products, where N_un is the number of unchanged
    bonds, N is the number of bonds in the reactants (i.e. unchanged plus lost), and
    N_add is the number if added bonds (i.e. total number of bonds in the products
    minus unchanged bonds).

    Args:
        reactants_graph: the graph of the reactants, Note this should be the combined
            graph for all molecules in the reactants.
        products_graph: the graph of the reactants, Note this should be the combined
            graph for all molecules in the reactants.
        num_unchanged_bonds: number of unchanged bonds in the reaction.
        num_lost_bonds: number of lost bonds in the reactants.
        num_added_bonds: number of added bonds in the products.
        num_global_nodes: number of global nodes to add. e.g. node holding global
            features of molecules.

    Returns:
        A graph representing the reaction.
    """

    # First add unchanged bonds and lost bonds from reactants
    rel = ("atom", "bond", "atom")
    src, dst = reactants_graph.edges(order="eid", etype=rel)
    a2a = [(u, v) for u, v in zip(src, dst)]

    # Then add added bonds from products
    src, dst, eid = products_graph.edges(form="all", order="eid", etype=rel)
    for u, v, e in zip(src, dst, eid):
        # e // 2 because two edges for each bond
        if torch.div(e, 2, rounding_mode="floor") >= num_unchanged_bonds:
            a2a.append((u, v))

    num_atoms = reactants_graph.num_nodes("atom")
    edges_dict = {("atom", "bond", "atom"): a2a}
    num_nodes_dict = {"atom": num_atoms}

    # global nodes
    if num_global_nodes > 0:
        a2v = []
        v2a = []
        for a in range(num_atoms):
            for v in range(num_global_nodes):
                a2v.append([a, v])
                v2a.append([v, a])

        edges_dict[("atom", "a2g", "global")] = a2v
        edges_dict[("global", "g2a", "atom")] = v2a
        num_nodes_dict["global"] = num_global_nodes

    g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)

    return g


def build_graph_and_featurize_reaction(
    mode,
    reaction: Reaction,
    atom_featurizer: AtomFeaturizer,
    bond_featurizer: BondFeaturizer,
    atom_speices,
    global_featurizer: Optional[GlobalFeaturizer] = None,
    num_global_nodes: int = 0,
    build_reaction_graph: bool = True,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, Union[dgl.DGLGraph, None]]:

    def featurize_one_mol(m: Molecule):
        g = mol_to_graph(m, num_global_nodes)

        rdkit_mol = m.rdkit_mol

        atom_feats = atom_featurizer(rdkit_mol, atom_speices)
        g.ndata['atom'] = atom_feats
        g = dgl.add_self_loop(g)
        return g

    try:
        reactant_graphs = [featurize_one_mol(m) for m in reaction.reactants]
        product_graphs = [featurize_one_mol(m) for m in reaction.products]

        if mode == 'train':
            return reactant_graphs, product_graphs

        else:
            # combine small graphs to form one big graph for reactants and products
            if len(reactant_graphs) > 1:
                reactants_g = combine_graphs(reactant_graphs)
            else:
                reactants_g = reactant_graphs[0]

            if len(product_graphs) > 1:
                products_g = combine_graphs(product_graphs)
            else:
                products_g = product_graphs[0]

            return reactants_g, products_g

    except Exception as e:
        raise RuntimeError(
            f"Cannot build graph and featurize for reaction: {reaction.id}"
        ) from e

    return reactant_graphs, product_graphs
