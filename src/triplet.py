import pandas as pd
import numpy as np
from rdkit import Chem
from reaction import Reaction
from typing import Any, Dict, List, Optional, Tuple, Union
from featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
class Triplet():
    def __init__(
        self,
        reaction,
        functional_group_smarts_filenames,
        reaction_center_mode: str = "functional_group",
    ):
        self.reaction = reaction
        self.reaction_center_mode = reaction_center_mode
        self.functional_group_smarts_filenames = functional_group_smarts_filenames
        filename = functional_group_smarts_filenames
        if not isinstance(filename, list):
            filename = [filename]
        dfs = [pd.read_csv(f, sep="\t") for f in filename]
        df = pd.concat(dfs)
        self.functional_groups = [Chem.MolFromSmarts(m) for m in df["smarts"]]

    def __call__(self, reactant_graphs, product_graphs):
        return self.reaction.get_reaction_center_atom_functional_group(self.functional_groups, reactant_graphs, product_graphs)