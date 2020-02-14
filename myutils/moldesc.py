from rdkit import Chem
from rdkit.Chem import AllChem, MolSurf
from random import random
from numpy import zeros, array, identity, diag
from multiprocessing import Pool

p_table = Chem.GetPeriodicTable()

electronegativity = {
   'H':  2.20 ,   
   'Li':    0.98 ,   
   'Be':    1.57 ,   
   'B':  2.04 ,   
   'C':  2.55 ,   
   'N':  3.04 ,   
   'O':  3.44 ,   
   'F':  3.98 ,   
   'Na':    0.93 ,   
   'Mg':    1.31 ,   
   'Al':    1.61 ,   
   'Si':    1.90 ,   
   'P':  2.19 ,   
   'S':  2.59 ,   
   'Cl':    3.16 ,   
   'K':  0.82 ,   
   'Ca':    1.00 ,   
   'Sc':    1.36 ,   
   'Ti':    1.54 ,   
   'V':  1.63 ,   
   'Cr':    1.66 ,   
   'Mn':    1.55 ,   
   'Fe':    1.83 ,   
   'Co':    1.88 ,   
   'Ni':    1.91 ,   
   'Cu':    1.90 ,   
   'Zn':    1.65 ,   
   'Ga':    1.81 ,   
   'Ge':    2.01 ,   
   'As':    2.18 ,   
   'Se':    2.55 ,   
   'Br':    2.96 ,   
   'Kr':    3.00 ,   
   'Rb':    0.82 ,   
   'Sr':    0.95 ,   
   'Y':  1.22 ,   
   'Zr':    1.33 ,   
   'Nb':    1.6  ,   
   'Mo':    2.16 ,   
   'Tc':    1.9  ,   
   'Ru':    2.2  ,   
   'Rh':    2.28 ,   
   'Pd':    2.20 ,   
   'Ag':    1.93 ,   
   'Cd':    1.69 ,   
   'In':    1.78 ,   
   'Sn':    1.96 ,   
   'Sb':    2.05 ,   
   'Te':    2.1  ,   
   'I':  2.66 ,   
   'Xe':    2.6  ,   
   'Cs':    0.79 ,   
   'Ba':    0.89 ,   
   'La':    1.10 ,   
   'Ce':    1.12 ,   
   'Pr':    1.13 ,   
   'Nd':    1.14 ,   
   'Sm':    1.17 ,   
   'Gd':    1.20 ,   
   'Dy':    1.22 ,   
   'Ho':    1.23 ,   
   'Er':    1.24 ,   
   'Tm':    1.25 ,   
   'Lu':    1.27 ,   
   'Hf':    1.3  ,   
   'Ta':    1.5  ,   
   'W':  2.36 ,   
   'Re':    1.9  ,   
   'Os':    2.2  ,   
   'Ir':    2.20 ,   
   'Pt':    2.28 ,   
   'Au':    2.54 ,   
   'Hg':    2.00 ,   
   'Tl':    1.62 ,   
   'Pb':    2.33 ,   
   'Bi':    2.02 ,   
   'Po':    2.0  ,   
   'At':    2.2  ,   
   'Ra':    0.9  ,   
   'Ac':    1.1  ,   
   'Th':    1.3  ,   
   'Pa':    1.5  ,   
   'U':  1.38 ,   
   'Np':    1.36 ,   
   'Pu':    1.28 ,   
   'Am':    1.3  ,   
   'Cm':    1.3  ,   
   'Bk':    1.3  ,   
   'Cf':    1.3  ,   
   'Es':    1.3  ,   
   'Fm':    1.3  ,   
   'Md':    1.3  }


class RandomDescriptors():
   def __init__(self, key='Lv',target_element='O', use_elements=True, size=8, keepHs=False ):
      self.key = key
      self.target_element = target_element
      self.size = size
      self.use_elements = use_elements
      self.keepHs=keepHs
      self.memory={}


   def _get_random_descriptor_of_element(self, element):
      if element in self.memory:
         return self.memory[element]
      else:
         vector = [random() for _ in range(self.size)]
         self.memory[element] = vector
         return vector


   def make_random_descriptors(self, smiles):
      mol = Chem.MolFromSmiles(smiles.replace('['+self.key, '['+self.target_element))
      result = []
      for Atom in mol.GetAtoms():
         idx=Atom.GetIdx()
         S = Atom.GetSymbol()
         NH = Atom.GetTotalNumHs()
         if self.use_elements:
            vector=self._get_random_descriptor_of_element(S)[:]
         else:
            vector=[random() for _ in range(self.size)]
         if self.keepHs:
            vector+=[float(NH)]
         result.append(vector)
      return result

   def __call__(self, smiles):
      return self.make_random_descriptors(smiles)


def _check_non_diagonal_zeros(matrix):
 N,M = matrix.shape
 assert N==M
 for i in range(N):
   for j in range(i+1,N):
      if matrix[i,j]==0:
         return True
 return False


def adjacency_to_distance(adj_matrix, min_dist_to_include=1, factor=1.0, max_dist_to_include=10):
   N,M = adj_matrix.shape
   assert N==M
   dist = zeros((N,N))
   b = adj_matrix[:,:]
   I=1.0
   
   while(_check_non_diagonal_zeros(dist) and I<=max_dist_to_include):
     for i in range(len(b)):
       for j in range(i+1,len(b)):
          if dist[i,j]==0 and b[i,j]>=min_dist_to_include:
            if factor==1.0:
               to_set = I
            else:
               to_set = I**factor
            dist[i,j] = to_set
            dist[j,i] = to_set
     I+=1.0
     b=b.dot(adj_matrix)
   return dist


def featurize_bond(bond, use_polarization=False):
    aromaticity = float(bond.GetIsAromatic())
    order = bond.GetBondTypeAsDouble()
    data= [order, aromaticity]
    begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
    if use_polarization:
       atom1, atom2 = begin_atom.GetSymbol(), end_atom.GetSymbol()
       el1, el2 = [electronegativity[x] for x in [atom1,atom2]]
       data.append(abs(el1-el2))
    return data, begin_atom.GetIdx(), end_atom.GetIdx()
      

def get_bond_space_matrices(mol, use_polarization=False):
   '''Returns: bond features, bond_adj_mtx, bond2atom_mtx ( X: A = X.dot(B) )'''
   Nbonds = mol.GetNumBonds()
   Natoms = mol.GetNumAtoms()
   bond_features = []
   bond2atom = zeros((Natoms, Nbonds))
   for bi, bond in enumerate(mol.GetBonds()):
      F, beg_i, end_i = featurize_bond(bond, use_polarization)
      bond_features.append(F)
      bond2atom[beg_i,bi]=1
      bond2atom[end_i,bi]=1
   bond_adj = bond2atom.T.dot(bond2atom)
   bond_adj = bond_adj - diag(diag(bond_adj))
   return array(bond_features), bond_adj, bond2atom
   

def compute_Gasteiger_charges(smiles, idx=0, Gasteiger_iterations=200):
   mol = Chem.MolFromSmiles(smiles)
   mol_ionized = Chem.MolFromSmiles(smiles)
   atom_ionized = mol_ionized.GetAtomWithIdx(idx)
   atom = mol.GetAtomWithIdx(idx)
   numHs = atom.GetTotalNumHs()
   if numHs>0:
      atom_ionized.SetNoImplicit(1)
      atom_ionized.SetFormalCharge(-1)
      atom_ionized.SetNumExplicitHs(numHs-1)
      Chem.rdmolops.SanitizeMol(mol_ionized)
   try:
      Chem.rdPartialCharges.ComputeGasteigerCharges(mol, Gasteiger_iterations, True)
      Chem.rdPartialCharges.ComputeGasteigerCharges(mol_ionized, Gasteiger_iterations, True)
      q_in_neu = atom.GetDoubleProp('_GasteigerHCharge') + atom.GetDoubleProp('_GasteigerCharge')
      q_in_ion = atom_ionized.GetDoubleProp('_GasteigerHCharge') + atom_ionized.GetDoubleProp('_GasteigerCharge')
   except ValueError:
      q_in_neu, q_in_ion = 0.0, 0.0
   is_ion_aromatic = atom_ionized.GetIsAromatic()
   return {'q_neu': q_in_neu, 'q_ion':q_in_ion, 'is_aromatic':is_ion_aromatic}


def describe_atom(atom_object, use_formal_charge=False, use_Gasteiger=False):
   mol = atom_object.GetOwningMol()
   contribs = MolSurf._LabuteHelper(mol)
   idx = atom_object.GetIdx()
   code = {'SP':1, 'SP2':2, 'SP3':3,'UNSPECIFIED':-1, 'UNKNOWN':-1, 'S':0, 'SP3D':4, 'SP3D2':5}
   result = []
   symbol = atom_object.GetSymbol()
   result.append(atom_object.GetAtomicNum())
   try:
      one_hot = [0.0 for _ in range(7)]
      hib = code[atom_object.GetHybridization().name]
      one_hot[hib+1]=1.0
      #result+=one_hot
      result.append(hib)
      result.append(atom_object.GetTotalValence())
   except:
      print(Chem.MolToSmiles(mol, canonical=0),idx)
      raise
   result.append(max(atom_object.GetNumImplicitHs(), atom_object.GetNumExplicitHs()))
   result.append(p_table.GetNOuterElecs(symbol))
   result.append(electronegativity.get(symbol,0))
   result.append(float(atom_object.GetIsAromatic()))
   if use_formal_charge:
      result.append(atom_object.GetFormalCharge())
   if use_Gasteiger:
      q_in_neu = atom_object.GetDoubleProp('_GasteigerHCharge') + atom_object.GetDoubleProp('_GasteigerCharge')
      result.append(q_in_neu)
   result.append(contribs[idx+1])
   return result


def process_smiles(smiles, use_bond_orders=False, use_formal_charge=False, add_connections_to_aromatic_rings=False, use_Gasteiger=True):
   if type(smiles).__name__=='str':
      mol = Chem.MolFromSmiles(smiles)
   elif type(smiles).__name__=='Mol':
      mol = smiles
   else:
      raise TypeError('Unknown type')
   A  = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(float)
   if use_bond_orders:
      for bond in mol.GetBonds():
         order = bond.GetBondTypeAsDouble()
         if bond.GetIsAromatic():
            order=1.5
         idx_beg = bond.GetBeginAtomIdx()
         idx_end = bond.GetEndAtomIdx()
         A[idx_beg, idx_end]=order
         A[idx_end, idx_beg]=order
   if add_connections_to_aromatic_rings:
      rings = mol.GetRingInfo().AtomRings()
      for R in rings:
         if not all([mol.GetAtomWithIdx(xx).GetIsAromatic() for xx in R]):continue
         for xx, idx1 in enumerate(R):
            for idx2 in R[xx+1:]:
               order =0.5 if use_bond_orders else 1.0
               if A[idx1, idx2] == 0:
                  A[idx1, idx2] = order
                  A[idx2, idx1] = order
   if use_Gasteiger:
      try:
         Chem.rdPartialCharges.ComputeGasteigerCharges(mol, 200, True)
      except ValueError:
         for atom in mol.GetAtoms():
            atom.SetProp('_GasteigerCharge','0.0')
            atom.SetProp('_GasteigerHCharge','0.0')
      
   desc = [describe_atom(x, use_formal_charge=use_formal_charge, use_Gasteiger=use_Gasteiger) for x in mol.GetAtoms()]
   return array(desc), A


def process_smiles_compressed(smiles, use_bond_orders=False, use_formal_charge=False, add_connections_to_aromatic_rings=False, use_Gasteiger=True):
   if type(smiles).__name__=='str':
      mol = Chem.MolFromSmiles(smiles)
   elif type(smiles).__name__=='Mol':
      mol = smiles
   else:
      raise TypeError('Unknown type')
   A  = array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
   if use_Gasteiger:
      try:
         Chem.rdPartialCharges.ComputeGasteigerCharges(mol, 200, True)
      except ValueError:
         for atom in mol.GetAtoms():
            atom.SetProp('_GasteigerCharge','0.0')
            atom.SetProp('_GasteigerHCharge','0.0')
      
   desc = [describe_atom(x, use_formal_charge=use_formal_charge, use_Gasteiger=use_Gasteiger) for x in mol.GetAtoms()]
   return array(desc), A



def _get_max_size(matrices):
   shapes = array([x.shape for x in matrices])
   return tuple(shapes.max(axis=0))


def zero_pad(source_matrix, target_shape):
   source_shape = source_matrix.shape
   D = len(target_shape)

   assert D<=2 and D>0
   assert len(source_shape)==D
   assert all([source_shape[x]<=target_shape[x] for x in range(D)])

   if source_shape==target_shape:
      result = source_matrix
   else:
      result = zeros(target_shape)
      if D==2:
         N, M = source_shape
         result[:N,:M] = source_matrix[:N,:M]
      elif D==1:
         N, = source_shape 
         result[:N] = source_matrix[:N]

   return result


def process_smiles_set(smiles_set, smiles_config, threads=0):
   if threads==0:
      result = [process_smiles(x, **smiles_config) for x in smiles_set]
   else:
      p=Pool(threads)
      result = p.map( lambda x: process_smiles(x, **smiles_config), smiles_set)
   X, A =  list(zip(*result))
   L = array([x.shape[0] for x in A])
   X_target_shape = _get_max_size(X)
   A_target_shape = _get_max_size(A)
   X=[zero_pad(x, X_target_shape) for x in X]
   A=[zero_pad(x, A_target_shape) for x in A]
   return {'X':array(X), 'A':array(A), 'L':L}



def test():
   mol = 'c1ccccc1CC(=O)O'
   X, A = process_smiles(mol)
   D = adjacency_to_distance(A)
   for x in X:print( x)
   print( A)
   print('\nWith bond orders')
   X, A = process_smiles(mol, use_bond_orders=1)
   print( A)
   print('\nDistance Matrix')
   print( D)
   mol=Chem.MolFromSmiles(mol)
   bf, bad, b2a = get_bond_space_matrices(mol, 1, 1)
   print('\nBond features\n',bf)
   print('\nBond adjacency\n',bad)
   print('\nBond to atom transform\n',b2a)
   print('\nAtom adj reconstructed\n',b2a.dot(b2a.T))

if __name__=='__main__':
   test()
