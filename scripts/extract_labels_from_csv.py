import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str)
parser.add_argument('--output_smiles', type=str, default=None, help='If present, smiles will be extracted and written here (if possible)')
parser.add_argument('--input', type=str)
parser.add_argument('--forbidden_column_names', type=str, nargs='+', default=['smiles'])
parser.add_argument('--nan_to_neg1', action='store_true', help='changes NAN to -1 (concerned in the case of missing data)')

args=parser.parse_args()

import numpy as np
import pandas as pd
import gzip

data = pd.read_csv(args.input)
if not (args.output_smiles is None):
   if 'smiles' in data.columns:
      smiles = data['smiles'].values
   elif 'SMILES' in data.colums:
      smiles = data['SMILES'].values
   else:
      print('Cannot find smiles column')
      smiles=None
   if not (smiles is None):
      np.savetxt(args.output_smiles, smiles, fmt='%s')

if args.forbidden_column_names!=[]:
   data = data.drop(args.forbidden_column_names, axis=1)

values = data.values
if args.nan_to_neg1:
   values = np.where(np.isnan(values), -1, values)

with gzip.open(args.output, 'wb') as f:
   np.save(f, values)

