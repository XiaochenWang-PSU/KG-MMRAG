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
from transformers import AutoTokenizer
import open_clip
from tqdm import tqdm

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
                                    lambda path: "/data/xiaochen/"+path
                                )))
        
        # Load KG and create mappings
        self.triplets = self._load_kg(kg_path)#[:100]
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
            # if row['Head'].startswith('I')
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
    def __init__(
        self,
        kg_path: str,
        image_map_path: str = "image_mapping.csv",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",  # or "laion2b_s34b_b79k"
        batch_size: int = 256,  # <-- smaller default to be safer on GPU
    ):
        super().__init__(kg_path, image_map_path)

        # ---- device ----
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = "cuda:1"
            else:
                self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.batch_size = batch_size

        # ---- open_clip model + preprocess + tokenizer ----
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # Store texts and *paths* instead of PIL images
        self.texts: List[str] = []
        self.image_paths: List[str] = []          # <--- changed
        self.image2idx: Dict[int, int] = {}

        # -------- build texts & image paths from triplets --------
        for idx, triplet in enumerate(tqdm(self.triplets, desc="Building texts & image paths")):
            # build text string
            text = ""
            if "image_" not in triplet.head_name.lower():
                text += triplet.head_name.lower() + " "
            text += triplet.relation.lower() + " " + triplet.tail_name.lower()
            self.texts.append(text)

            # attach image path if exists
            if (
                triplet.head in self.image_id_to_path
                and os.path.exists(self.image_id_to_path[triplet.head])
            ):
                self.image2idx[len(self.image_paths)] = idx
                self.image_paths.append(self.image_id_to_path[triplet.head])

        # -------- encode all texts & images with open_clip --------
        self.text_emb = self._encode_texts(self.texts)  # (N_nodes, D)

        if len(self.image_paths) > 0:
            self.image_emb = self._encode_images(self.image_paths)  # (N_images, D)
        else:
            # empty tensor with correct dim
            self.image_emb = torch.empty(
                0, self.text_emb.size(1), dtype=self.text_emb.dtype
            )

        # -------- build multimodal retrieval embeddings (avg text+image) --------
        self.retrieval_embeddings: List[torch.Tensor] = [[] for _ in range(len(self.texts))]

        # add image embeddings per triplet (if any)
        for i in range(len(self.image_emb)):
            node_idx = self.image2idx[i]
            self.retrieval_embeddings[node_idx].append(self.image_emb[i])

        # add text embeddings (always one per node)
        for i in range(len(self.text_emb)):
            self.retrieval_embeddings[i].append(self.text_emb[i])

        # average across modalities and normalize
        for i in range(len(self.retrieval_embeddings)):
            # each entry is a list of 1 or 2 vectors (text, [image])
            stacked = torch.stack(self.retrieval_embeddings[i], dim=0).mean(dim=0)
            self.retrieval_embeddings[i] = stacked

        self.retrieval_embeddings = torch.stack(self.retrieval_embeddings, dim=0)  # (N_nodes, D)
        self.retrieval_embeddings = torch.nn.functional.normalize(
            self.retrieval_embeddings, p=2, dim=1
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts with open_clip in batches.
        Returns embeddings on CPU.
        """
        all_embs = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            text_features = self.model.encode_text(tokens)
            all_embs.append(text_features.cpu())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def _encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of image paths with open_clip in batches.
        Images are loaded on the fly to avoid holding them all in RAM.
        Returns embeddings on CPU.
        """
        all_embs = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start:start + self.batch_size]

            pil_images = []
            for p in batch_paths:
                with Image.open(p) as im:
                    pil_images.append(im.convert("RGB"))

            # preprocess to tensors and stack
            img_tensors = [self.preprocess(im) for im in pil_images]
            img_batch = torch.stack(img_tensors, dim=0).to(self.device)

            img_features = self.model.encode_image(img_batch)
            all_embs.append(img_features.cpu())

            # free GPU memory between big batches if needed
            del img_batch, img_features
            torch.cuda.empty_cache()

        return torch.cat(all_embs, dim=0)

    # ----------------- public search API -----------------

    def search(self, sample: Dict[str, Any], k: int):
        # image
        if "image_path" in sample:
            with Image.open(sample["image_path"]) as im:
                query_image = im.convert("RGB")
        else:
            query_image = sample["image"]

        # text
        query_text = sample["question"].lower()

        with torch.no_grad():
            # encode query text
            tok = self.tokenizer([query_text]).to(self.device)        # (1,77)
            q_text_feat = self.model.encode_text(tok)                 # (1,D)
            q_text_feat = q_text_feat / q_text_feat.norm(dim=-1, keepdim=True)

            # encode query image
            q_img_tensor = self.preprocess(query_image).unsqueeze(0).to(self.device)
            q_img_feat = self.model.encode_image(q_img_tensor)        # (1,D)
            q_img_feat = q_img_feat / q_img_feat.norm(dim=-1, keepdim=True)

            # average text + image & normalize
            q_feat = (q_text_feat + q_img_feat) / 2.0                 # (1,D)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)       # (1,D)

        # compute cosine similarity via dot product (already L2-normalized)
        # retrieval_embeddings is on CPU; move query to same device
        q_feat_cpu = q_feat.cpu()
        scores = (self.retrieval_embeddings @ q_feat_cpu.T).flatten()  # (N_nodes,)

        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append(
                {
                    "rank": rank,
                    "index": i,
                    "item": self.triplets[i],
                    "score": float(s),
                }
            )
        return out
class SimpleTextRetriever(BaseRetriever):
    def __init__(self, kg_path: str, image_map_path: str = "image_mapping.csv",
                 model_name="clip-ViT-B-32", batch_size=32):
        super().__init__(kg_path, image_map_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.texts = []
        # mapping: index in self.texts / self.retrieval_embeddings -> index in self.triplets
        self.text_idx2triplet_idx = []

        for idx, triplet in enumerate(self.triplets):
            # deliberately exclude "image_" heads from ranking
            if "image_" in triplet.head_name.lower():
                continue

            text = (
                triplet.head_name.lower()
                + " "
                + triplet.relation.lower()
                + " "
                + triplet.tail_name.lower()
            )
            self.texts.append(text)
            self.text_idx2triplet_idx.append(idx)

        self.retrieval_embeddings = self.model.encode(
            self.texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        print(self.retrieval_embeddings.shape)

        # Normalize for cosine similarity via dot product
        self.retrieval_embeddings = torch.nn.functional.normalize(
            self.retrieval_embeddings, p=2, dim=1
        )

    def search(self, sample, k):
        query_text = sample["question"].lower()

        query_embedding = self.model.encode(
            [query_text],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        query_embedding = torch.nn.functional.normalize(
            query_embedding, p=2, dim=1
        )

        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()

        # guard in case k > #candidates
        k = min(k, scores.shape[0])

        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            # i indexes into self.texts / self.retrieval_embeddings
            triplet_idx = self.text_idx2triplet_idx[i]  # map back to original triplet index
            triplet = self.triplets[triplet_idx]

            out.append(
                {
                    "rank": rank,
                    "index": triplet_idx,  # index in the original triplet list
                    "item": triplet,
                    "score": float(s),
                }
            )

        return out



class RandomRetriever(BaseRetriever):
    def search(self, sample, k):
        random_idx = random.sample([i for i in range(len(self.triplets))], k)

        out = []
        for rank, i in enumerate(random_idx):
            out.append({
                "rank": rank + 1,
                "index": i,
                "item": self.triplets[i],
                "score": rank + 1,
            })
        
        return out
