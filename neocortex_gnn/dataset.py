import random
from pathlib import Path

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
import torch
from torch.utils.data import Dataset
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, GetDistanceMatrix
from rdkit.Chem.rdmolfiles import MolFromPDBFile

from neocortex_gnn.util import one_hot_encode_dataframe_column

random.seed(0)


class ReceptorAndOptimalMatchingSpheresDataset(Dataset):
    PDB_COLUMNS_TO_KEEP = [  # TODO: confirm this with Andrii
        "atom_name",
        "residue_name",
        "x_coord",
        "y_coord",
        "z_coord",
        "occupancy",
        "b_factor",
        "element_symbol",
    ]
    ONE_HOT_ENCODABLE_PDB_COLUMNS_TO_ALLOWED_VALUES_DICT = {
        "residue_name": [  # TODO: confirm that 20 standard amino acids are sufficient
            'ALA',
            'ARG',
            'ASN',
            'ASP',
            'CYS',
            'GLN',
            'GLU',
            'GLY',
            'HIS',
            'ILE',
            'LEU',
            'LYS',
            'MET',
            'PHE',
            'PRO',
            'SER',
            'THR',
            'TRP',
            'TYR',
            'VAL',
        ],
        "atom_name": [  # TODO: get exhaustive list
            'C',
            'CA',
            'CB',
            'CD',
            'CD1',
            'CD2',
            'CE',
            'CE1',
            'CE2',
            'CE3',
            'CG',
            'CG1',
            'CG2',
            'CH2',
            'CZ',
            'CZ2',
            'CZ3',
            'N',
            'ND1',
            'ND2',
            'NE',
            'NE1',
            'NE2',
            'NH1',
            'NH2',
            'NZ',
            'O',
            'OD1',
            'OD2',
            'OE1',
            'OE2',
            'OG',
            'OG1',
            'OH',
            'SD',
            'SG',
        ],
        "element_symbol": ['H', 'C', 'N', 'O', 'S'],  # TODO: get exhaustive list
    }
    #PDB_COLUMNS_TO_NORMALIZE = []  # TODO

    def __init__(self, keys, receptors_dir, matching_spheres_dir):
        self.keys = keys
        self.receptors_dir = Path(receptors_dir)
        self.matching_spheres_dir = Path(matching_spheres_dir)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #
        key = self.keys[idx]
        pdb_file_path = Path(self.receptors_dir.joinpath(f"{key}.pdb"))
        sph_file_path = Path(self.matching_spheres_dir.joinpath(f"{key}.sph").as_posix())

        # load PDB to dataframe
        df_pdb = PandasPdb().read_pdb(pdb_file_path.as_posix()).df["ATOM"]

        # remove irrelevant PDB columns
        df_pdb = df_pdb.drop([col for col in df_pdb.columns if col not in self.PDB_COLUMNS_TO_KEEP], axis=1)
        if set(df_pdb.columns) != set(self.PDB_COLUMNS_TO_KEEP):
            raise ValueError(f"PDB is missing at least one expected column.\n\tWitnessed columns: {df_pdb.columns}\n\tExpected columns: {self.PDB_COLUMNS_TO_KEEP}")

        # one-hot encode relevant PDB columns
        for column, allowed_values in self.ONE_HOT_ENCODABLE_PDB_COLUMNS_TO_ALLOWED_VALUES_DICT.items():
            df_pdb = one_hot_encode_dataframe_column(df_pdb, column, allowed_values)

        # normalize relevant PDB columns
        # TODO

        # load PDB as Mol
        mol = MolFromPDBFile(pdb_file_path.as_posix())

        # get adjacency matrix
        adjacency_matrix = GetAdjacencyMatrix(mol)

        # get distance matrix
        distance_matrix = GetDistanceMatrix(mol)

        # load SPH to dataframe
        sph_records = []
        with open(sph_file_path.as_posix(), 'r') as f:
            valid = False
            for i, line in enumerate(f.readlines()):
                if line.startswith("cluster"):
                    valid = True
                    continue
                if valid:
                    sph_records.append(line.strip().split())
        sph_records = np.array(sph_records, dtype=float)
        df_sph = pd.DataFrame.from_records(sph_records)

        # keep only relevant SPH columns
        df_sph.columns = ['?1', 'x', 'y', 'z', '?2', '?3', '?4', '?5']
        for column in df_sph.columns:
            if column not in ['x', 'y', 'z']:
                df_sph.drop(column, axis=1, inplace=True)

        # if n1+n2 > 300 : return None
        sample = {
            'P': df_pdb.apply(pd.to_numeric).to_numpy(),
            'A': adjacency_matrix,
            'D': distance_matrix,
            'S': np.reshape(df_sph.apply(pd.to_numeric).to_numpy(), (45*3,)),
            'key': key,
        }

        return sample


def collate_fn(batch):
    #
    max_natoms = max([len(item['P']) for item in batch if item is not None])

    #
    P = np.zeros((len(batch), max_natoms, 65))
    A = np.zeros((len(batch), max_natoms, max_natoms))
    D = np.zeros((len(batch), max_natoms, max_natoms))
    S = np.zeros((len(batch), 45*3))
    keys = []
    for i in range(len(batch)):
        natom = len(batch[i]['P'])
        P[i, :natom] = batch[i]['P']
        A[i, :natom, :natom] = batch[i]['A']
        D[i, :natom, :natom] = batch[i]['D']
        S[i] = batch[i]['S']
        keys.append(batch[i]['key'])

    #
    P = torch.from_numpy(P).float()
    A = torch.from_numpy(A).float()
    D = torch.from_numpy(D).float()
    S = torch.from_numpy(S).float()

    return P, A, D, S, keys

