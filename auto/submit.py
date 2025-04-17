#!/usr/bin/env python3
import sys, urllib, re, requests, cloudpickle
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
from catboost import CatBoostRegressor
import time

start_time = time.time()

#  Download Helper 
def get_sharepoint(url, fname):
    '''Fetch a world-readable SharePoint file shared via URL'''
    s = requests.Session()
    r = s.get(url)
    if not r:
        return False
    m = re.search(r'FileRef":\s+"(.*?)"', r.text)
    if not m:
        return False
    path = "'" + m.group(1).replace('\\u002f', '/') + "'"
    path = urllib.parse.quote_plus(path, safe='').replace('.', '%2E').replace('_', '%5F')
    comp = url.split('/')
    durl = f'{comp[0]}//{comp[2]}/{comp[5]}/{comp[6]}/_api/web/GetFileByServerRelativePath(DecodedUrl=@a1)/OpenBinaryStream?@a1={path}'
    r = s.get(durl, stream=True)
    if not r:
        return False
    with open(fname, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    return True

#  Download all files 
get_sharepoint("https://pitt-my.sharepoint.com/:u:/g/personal/daa248_pitt_edu/EXax2uFOUN9Cqdc1RLIGLUMBr4M_RsN5IaAJZLC0KBmt9w?e=2O9nVl", "model.pkl")
get_sharepoint("https://pitt-my.sharepoint.com/:t:/g/personal/daa248_pitt_edu/EdUlcAZ2gdFGnrqTCCUrQY0BUMEKKTWg-V1YF0BDe0AN1g?e=i7lyLt", "kmer_vocab.txt")
get_sharepoint("https://pitt-my.sharepoint.com/:t:/g/personal/daa248_pitt_edu/EY_LoADXgS5CvrlY-4JtPnUBeKodPxIAWb2GUKUQbEy_Kg?e=FANqjc", "kinase_seqs.txt")

#  Load model 
with open("model.pkl", "rb") as f:
    model = cloudpickle.load(f)

#  Load vocabulary and sequence map 
with open("kmer_vocab.txt") as f:
    vocab = [line.strip() for line in f]

kmer_to_index = {kmer: i for i, kmer in enumerate(vocab)}
k = len(vocab[0])

seq_map = {}
with open("kinase_seqs.txt") as f:
    for line in f:
        if line.strip():
            uniprot, seq = line.strip().split()
            seq_map[uniprot] = seq

#  Featurization 
def featurize(smile, uniprot):
    # Morgan fingerprint
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = GetMorganGenerator(radius=2, fpSize=2048)
    fp_array = np.array(fp.GetFingerprint(mol), dtype=np.float32)

    # K-mer count
    kmer_vec = np.zeros(len(kmer_to_index))
    seq = seq_map.get(uniprot)
    if seq:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in kmer_to_index:
                kmer_vec[kmer_to_index[kmer]] += 1
    return np.concatenate([fp_array, kmer_vec])

#  Run predictions 
infile = open(sys.argv[1])
outfile = open(sys.argv[2], "wt")

header = infile.readline()
outfile.write(header.strip() + " pChEMBL\n")

for line in infile:
    smile, uniprot = line.strip().split()
    x = featurize(smile, uniprot)
    if x is None:
        val = 6.0  # fallback if invalid SMILES
    else:
        val = float(model.predict(np.array([x]))[0])
    outfile.write(f"{smile} {uniprot} {val:.4f}\n")

print('Time to run:', time.time() - start_time)
