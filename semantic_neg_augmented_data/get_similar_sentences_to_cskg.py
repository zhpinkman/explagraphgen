import spacy
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import joblib
import torch

kg_df = pd.read_csv("../cskg_star.tsv", sep="\t", error_bad_lines=False)

all_kg_sentences = []
print(kg_df.shape)
for _, row in tqdm(kg_df.iterrows()):
    if str(row['sentence']) != 'nan':
        all_kg_sentences.append(row['sentence'])
    else:
        all_kg_sentences.append(f"{row['node1;label']} {row['relation;label']} {row['node2;label']}")
        
        

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

model.to(device)

embeddings = model.encode(all_kg_sentences, show_progress_bar=True, device=device)


joblib.dump(embeddings, "all_kg_sentences_embeddings.pkl")



with open('train.source', 'r', encoding='utf-8-sig') as f:
    train_source = f.read().splitlines()
    
    
unique_sentences = list(set(train_source))
len(unique_sentences)


beliefs = []
arguments = []
for sent in unique_sentences:
    all_seps = [m.start() for m in re.finditer('</s>', sent)]
    beliefs.append(sent[7:all_seps[0]])
    arguments.append(sent[all_seps[0]+14:all_seps[1]])
    
    
beliefs_embeddings = model.encode(beliefs, show_progress_bar=True, device=device)
joblib.dump(beliefs_embeddings, "beliefs_embeddings.pkl")

arguments_embeddings = model.encode(arguments, show_progress_bar=True, device=device)
joblib.dump(arguments_embeddings, "arguments_embeddings.pkl")



