#!/usr/bin/env python3
import sys, urllib, re, requests, cloudpickle
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
import xgboost as xgb
import time

start_time = time.time()

# Download helper
def get_sharepoint(url, fname):
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

# Download required files
get_sharepoint("https://pitt-my.sharepoint.com/:u:/g/personal/daa248_pitt_edu/EfCZ-V2GcCVNp6L8Vms9DzoB-Ny2O-ldBVwzvk1X9sR84A?e=VQWaxY", "model.pkl")
get_sharepoint("https://pitt-my.sharepoint.com/:t:/g/personal/daa248_pitt_edu/EY_LoADXgS5CvrlY-4JtPnUBeKodPxIAWb2GUKUQbEy_Kg?e=FANqjc", "kinase_seqs.txt")
get_sharepoint("https://pitt-my.sharepoint.com/:u:/g/personal/daa248_pitt_edu/EbgpSLAyFAhEmaPEabv26_0Bfne6Ts287b-kc0ctlDuHhw?e=bUkCxH", "esm2_embeddings.npy")

# Load model
with open("model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Load sequences
seq_map = {}
with open("kinase_seqs.txt") as f:
    for line in f:
        if line.strip():
            uniprot, seq = line.strip().split()
            seq_map[uniprot] = seq

# Load ESM-2 embeddings
embedding_dict = np.load("esm2_embeddings.npy", allow_pickle=True).item()

# Featurize: ECFP + ESM embedding
def featurize(smile, uniprot):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = GetMorganGenerator(radius=2, fpSize=2048)
    ecfp_vec = np.array(fp.GetFingerprint(mol), dtype=np.float32)

    if uniprot not in embedding_dict:
        return None
    esm_vec = embedding_dict[uniprot]

    return np.concatenate([ecfp_vec, esm_vec])

# Run predictions
infile = open(sys.argv[1])
outfile = open(sys.argv[2], "wt")

header = infile.readline()
outfile.write(header.strip() + " pChEMBL\n")

for line in infile:
    smile, uniprot = line.strip().split()
    x = featurize(smile, uniprot)
    if x is None:
        val = 6.0
    else:
        val = float(model.predict(xgb.DMatrix(np.array([x])))[0])
    outfile.write(f"{smile} {uniprot} {val:.4f}\n")

print('Time to run:', time.time() - start_time)
