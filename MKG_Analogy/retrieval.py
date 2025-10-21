import os
from typing import List, Dict, Any, Tuple
from PIL import Image
from utils import *
import torch
import numpy as np
import numpy
import torch
import random
print(torch.__version__)
from sentence_transformers import SentenceTransformer

class SimpleRetriever:
    # In this retrieval, for every triplet (head, relation, tail), we will take average 
    def __init__(self, retrieval_dataset, entity2text, relation2text, model_name, batch_size=32):
        # retrieval_dataset = [[h1, r1, t1], [h2, r2, t2], ...]

        # Initialization
        self.retrieval_dataset = retrieval_dataset
        
        self.text = []
        self.image = []
        
        self.text2retrieval = {}
        self.image2retrieval = {}
        
        self.entity2text = entity2text

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        for idx, triplet in enumerate(self.retrieval_dataset):
            head, relation, tail = triplet
            if head in entity2text:
                self.text2retrieval[len(self.text)] = idx
                self.text.append(entity2text[head])

            if tail in entity2text:
                self.text2retrieval[len(self.text)]= idx
                self.text.append(entity2text[tail])

            if relation in relation2text:
                self.text2retrieval[len(self.text)] = idx
                self.text.append(relation2text[relation])

            if first_jpg_path(head, "images_subset_kg"):
                self.image2retrieval[len(self.image)] = idx 
                with Image.open(first_jpg_path(head, "images_subset_kg")) as im:    
                    self.image.append(im.convert("RGB"))

            if first_jpg_path(tail, "images_subset_kg"):
                self.image2retrieval[len(self.image)] = idx
                with Image.open(first_jpg_path(tail, "images_subset_kg")) as im:
                    self.image.append(im.convert("RGB"))
        
        # Encode text and image to embedding
        self.image_emb = self.model.encode(self.image, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        self.text_emb = self.model.encode(self.text, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)

        # Take Average
        self.retrieval_embeddings = [[] for i in range(len(retrieval_dataset))]
        
        for i in range(len(self.image_emb)):
            self.retrieval_embeddings[self.image2retrieval[i]].append(self.image_emb[i])
        
        for i in range(len(self.text_emb)):
            self.retrieval_embeddings[self.text2retrieval[i]].append(self.text_emb[i])

        for i in range(len(retrieval_dataset)):
            self.retrieval_embeddings[i] = torch.stack(self.retrieval_embeddings[i], dim = 0).mean(dim = 0)

        self.retrieval_embeddings = torch.stack(self.retrieval_embeddings, dim = 0)

        # Normalize for cosine similarity via dot product
        self.retrieval_embeddings = torch.nn.functional.normalize(self.retrieval_embeddings, p=2, dim=1)

    def search(self, query, k, mode):
        # query: [head, tail, question]. Do not have relation since MARS do not allow to provide relation to model.
        # mode is defined as follow
        #   mode 0: (T1, T2) -> (I1, ?)
        #   mode 1: (I1, I2) -> (T1, ?)
        #   mode 2: (I1, T1) -> (I2, ?)

        query_embeddings = []

        head, tail, question = query

        if mode == 0:
            query_embeddings.append(self.model.encode([self.entity2text[head]], convert_to_tensor=True, show_progress_bar=False))
            query_embeddings.append(self.model.encode([self.entity2text[tail]], convert_to_tensor=True, show_progress_bar=False))
            with Image.open(first_jpg_path(question, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im], convert_to_tensor=True, show_progress_bar=False))

        if mode == 1:
            with Image.open(first_jpg_path(head, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im], convert_to_tensor=True, show_progress_bar=False))
            with Image.open(first_jpg_path(tail, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im], convert_to_tensor=True, show_progress_bar=False))
            query_embeddings.append(self.model.encode([self.entity2text[question]], convert_to_tensor=True, show_progress_bar=False))

        if mode == 2:
            with Image.open(first_jpg_path(head, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im], convert_to_tensor=True, show_progress_bar=False))
            query_embeddings.append(self.model.encode([self.entity2text[tail]], convert_to_tensor=True, show_progress_bar=False))
            with Image.open(first_jpg_path(question, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im], convert_to_tensor=True, show_progress_bar=False))

        query_embedding = torch.stack(query_embeddings, dim = 0).mean(dim = 0)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)  
        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()
        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append({
                "rank": rank,
                "index": i,
                "item": self.retrieval_dataset[i],
                "score": float(s),
            })
        
        return out

triplets = load_triplets("dataset/MarKG/wiki_tuple_ids.txt")
triplets = random.sample(triplets, 10)

entity2text = read_txt("dataset/MarKG/entity2text.txt")
relation2text = read_txt("dataset/MarKG/relation2text.txt")

query =  ["Q65386997", "Q44705078", "Q271960"] # first question in test.json file
retriever = SimpleRetriever(triplets, entity2text, relation2text, "clip-ViT-B-32")
print(retriever.search(query, 5, 0))