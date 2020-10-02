import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--output_core', type=str)
parser.add_argument('--descriptor', type=str, choices=['rdkit','ecfp4','mol2vec'], default='rdkit')
parser.add_argument('--mol2vec_model_pkl', type=str, default=None)
parser.add_argument('input', type=str)
args=parser.parse_args()

assert not (args.descriptor=='mol2vec' and (args.mol2vec_model_pkl is None)),\
 'With Mol2Vec, you should supply pickle with trained Mol2Vec model (see Mol2Vec docs)'

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec
import numpy as np
import pickle
import gzip
import time

def gzsave(name, array):
   with gzip.open(name, 'wb') as f:
      np.save(f, array)

keys = [x[0] for x in Descriptors.descList]
desc = dict(Descriptors.descList)
keys.sort()


def vectorize_rdkit(smiles, mol=None):
   if mol is None:
      mol = Chem.MolFromSmiles(smiles)
   return [desc[x](mol) for x in keys]


def vectorize_morgan(smiles, mol=None, r=2, use_features=False, nbits=2048):
   if mol is None:
      mol = Chem.MolFromSmiles(smiles)
   f = AllChem.GetMorganFingerprint(mol, r, useFeatures=use_features)
   elements = f.GetNonzeroElements()
   result = np.zeros(nbits)
   for tag in elements:
      idx = int(tag%nbits)
      result[idx] = elements[tag]
   return result


simple_processors = {'rdkit':vectorize_rdkit, 'ecfp4':vectorize_morgan}

def smiles2sentence(smiles):
   mol=Chem.MolFromSmiles(smiles)
   sentence = mol2alt_sentence(mol,1)
   return sentence

#===========================
beg=time.time()

#load smiles from smiles file
with open(args.input, 'r') as f:
   smiles_list=f.readlines()

if args.descriptor=='mol2vec':
   model = word2vec.Word2Vec.load(args.mol2vec_model_pkl)
   sentences=[smiles2sencence(x) for x in smiles_list]
   X = sentences2vec(sentences, model, unseen='UNK').astype(float32)
else:
   processor = simple_processors[args.descriptor]
   X = [processor(x) for x in smiles_list]

end=time.time()
print('processed array of %i compounds in %f seconds'%(len(X), end-beg))

X=np.nan_to_num(np.array(X).astype(np.float32))
mean = np.nan_to_num(X.mean(axis=0))
std = np.nan_to_num(X.std(axis=0))
non_zero_idx = np.where(std>0)[0]

gzsave(args.output_core+'.npz', X)
gzsave(args.output_core+'_mu.npz', mean)
gzsave(args.output_core+'_std.npz', std)
gzsave(args.output_core+'_idx.npz', non_zero_idx)
