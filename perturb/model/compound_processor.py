import gzip
import pickle
from rdkit import Chem

allowable_features = {
    'possible_atomic_num_list' : list(range(0, 118)) + ['<unk>'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, '<unk>'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, '<unk>'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, '<unk>'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, '<unk>'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', '<unk>'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        '<unk>'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

SMILES_TO_GRAPH={}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_features(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            # safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()), 
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        # allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list'],
        allowable_features['possible_atomic_num_list']
        ]))

def preprocess_smiles(smiles, cleanup_chirality = False):
    """
    clean up smiles using chembl_structure_pipeline
    :para smiles: str smiles
    :para cleanup_chirality: bool, whether or not to remove chirality
    :return: cleaned smiles by chembl_structure_pipeline
    """
    # 1 if the input smiles can be properly converted, 0 if there is an error
    flag = True
    
    try:
        # get mol object
        mol = Chem.MolFromSmiles(smiles)
        # standardize mol
        mol_std = standardize_mol(mol)
        # get parent
        mol_par = get_parent_mol(mol_std)[0]
        # canonicalize SMILES, remove chirality if required
        if cleanup_chirality:
            smiles = Chem.MolToSmiles(mol_par, isomericSmiles = False)
        else:
            smiles = Chem.MolToSmiles(mol_par)        
        mol = Chem.MolFromSmiles(smiles)
        smiles_canonicalized = Chem.MolToSmiles(mol)
        
    except:
        smiles_canonicalized = smiles
        print(f"Error for SMILES: {smiles}")
        flag = False
        
    return smiles_canonicalized, flag

def canonicalize_molecule(molecule, addH=True):
    if addH:
        molecule = Chem.AddHs(molecule)  # add back all hydrogen atoms
    order    = Chem.CanonicalRankAtoms(molecule)  # to get a canonical form here
    molecule = Chem.RenumberAtoms(molecule, order)
    return molecule

class gpickle(object):
    """
    A pickle class with gzip enabled
    """
    @staticmethod
    def dump(data, filename, compresslevel=9, protocol=4):
        with gzip.open(filename, mode='wb', compresslevel=compresslevel) as f:
            pickle.dump(data, chunked_byte_writer(f), protocol=protocol)
            f.close()

    @staticmethod
    def load(filename):
        """
        The chunked read mechanism here is for by-passing the bug in gzip/zlib library: when
        data length exceeds unsigned int limit, gzip/zlib will break
        :param filename:
        :return:
        """
        buf = b''
        chunk = b'NULL'
        with gzip.open(filename, mode='rb') as f:
            while len(chunk) > 0:
                chunk = f.read(429496729)
                buf += chunk
        data = pickle.loads(buf)
        return data

    @staticmethod
    def loads(zipped_bytes):
        bytes = gzip.decompress(zipped_bytes)
        return pickle.loads(bytes)

    @staticmethod
    def dumps(data, compresslevel=9, protocol=4):
        zipped_bytes = gzip.compress(pickle.dumps(data, protocol=protocol), compresslevel=compresslevel)
        return zipped_bytes
