import hashlib
import re

import numpy as np


class FastEmbedEmbedder:
    def __init__(self, model_name):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.dim = len(next(iter(self.model.embed(["test"]))))

    def embed(self, texts):
        return [v.tolist() for v in self.model.embed(texts)]


class HashEmbedder:
    # simpler bag of words embedder, braucht keinen model download -> gut für tests
    def __init__(self, dim=256):
        self.dim = dim

    def embed(self, texts):
        result = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little")
                vec[h % self.dim] += 1
            norm = np.linalg.norm(vec)
            if norm:
                vec = vec / norm
            result.append(vec.tolist())
        return result


def build_embedder(kind, model_name):
    if kind == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(model_name)
