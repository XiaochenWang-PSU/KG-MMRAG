import os
from typing import List, Dict, Any, Tuple
from PIL import Image
from utils import *
import torch
import numpy as np
import numpy
import torch
import random
from sentence_transformers import SentenceTransformer
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

@dataclass
class KGTriplet:
    head: str  # Head ID
    head_name: str  # Head name/description
    relation: str  # type of relation
    tail: str  # Tail ID
    tail_name: str  # Tail name/description


class BaseRetriever:
    def __init__(self, kg_path: str, image_map_path: str = "image_mapping.csv"):    
        # Load image mapping
        df_image_map = pd.read_csv(image_map_path)
        self.image_id_to_path = dict(zip(df_image_map['IID'], 
                                df_image_map['Image_Path'].apply(
                                    lambda path: "MedMKG_huggingface/images/"+path.split("/")[-1]
                                )))
        
        # Load KG and create mappings
        self.triplets = self._load_kg(kg_path)
        self._create_mappings()
       
    def _load_kg(self, kg_path: str) -> List[KGTriplet]:
        if not Path(kg_path).exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {kg_path}")
            
        df = pd.read_csv(kg_path)
        return [
            KGTriplet(
                head=str(row['Head']),
                head_name=str(row['Head_Name']),
                relation=str(row['Relation']),
                tail=str(row['Tail']),
                tail_name=str(row['Tail_Name'])
            )
            for _, row in df.iterrows()
            if row['Head'].startswith('I')
        ]

    def _create_mappings(self):
        self.id_to_name = {}
        self.name_to_ids = {}
        self.tail_to_heads = {}
        self.head_to_tails = {}
        
        for triplet in self.triplets:
            # ID-name mappings
            self.id_to_name[triplet.head] = triplet.head_name
            self.id_to_name[triplet.tail] = triplet.tail_name
            
            for name, id_ in [(triplet.head_name, triplet.head), 
                            (triplet.tail_name, triplet.tail)]:
                if name not in self.name_to_ids:
                    self.name_to_ids[name] = set()
                self.name_to_ids[name].add(id_)
            
            # Relation mappings
            for source, target, mapping in [
                (triplet.tail, triplet.head, self.tail_to_heads),
                (triplet.head, triplet.tail, self.head_to_tails)
            ]:
                if source not in mapping:
                    mapping[source] = {}
                if triplet.relation not in mapping[source]:
                    mapping[source][triplet.relation] = set()
                mapping[source][triplet.relation].add(target)


class SimpleMultimodalRetriever(BaseRetriever):
    def __init__(self, kg_path: str, image_map_path: str = "image_mapping.csv", model_name="clip-ViT-B-32", batch_size=32):
        super().__init__(kg_path, image_map_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.texts = []
        self.images = []

        self.image2idx = {}
        
        for idx, triplet in enumerate(self.triplets[:100]):   
            text = ""

            if "image_" not in triplet.head_name.lower():
                text += triplet.head_name.lower()
                text += " "
            
            text += triplet.relation.lower() + " " + triplet.tail_name.lower()
            self.texts.append(text)

            if triplet.head in self.image_id_to_path and os.path.exists(self.image_id_to_path[triplet.head]):
                self.image2idx[len(self.images)] = idx
                with Image.open(self.image_id_to_path[triplet.head]) as im:    
                    self.images.append(im.convert("RGB"))
                
        self.image_emb = self.model.encode(self.images, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        self.text_emb = self.model.encode(self.texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)

        # Take Average
        self.retrieval_embeddings = [[] for i in range(len(self.texts))]
        
        for i in range(len(self.image_emb)):
            self.retrieval_embeddings[self.image2idx[i]].append(self.image_emb[i])
        
        for i in range(len(self.text_emb)):
            self.retrieval_embeddings[i].append(self.text_emb[i])

        for i in range(len(self.retrieval_embeddings)):
            self.retrieval_embeddings[i] = torch.stack(self.retrieval_embeddings[i], dim = 0).mean(dim = 0)

        self.retrieval_embeddings = torch.stack(self.retrieval_embeddings, dim = 0)

        # Normalize for cosine similarity via dot product
        self.retrieval_embeddings = torch.nn.functional.normalize(self.retrieval_embeddings, p=2, dim=1)

    def search(self, sample, k):
        if "image_path" in sample:
            with Image.open(sample["image_path"]) as im:
                query_image = im.convert("RGB")
        else:
            query_image = sample["image"]
        query_text = sample["question"].lower()

        query_embeddings = []
        query_embeddings.append(self.model.encode([query_text], convert_to_tensor=True, show_progress_bar=False))
        query_embeddings.append(self.model.encode([query_image], convert_to_tensor=True, show_progress_bar=False))
        query_embedding = torch.stack(query_embeddings, dim = 0).mean(dim = 0)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)  
        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()
        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append({
                "rank": rank,
                "index": i,
                "item": self.triplets[i],
                "score": float(s),
            })
        
        return out