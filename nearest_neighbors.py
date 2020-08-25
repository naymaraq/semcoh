import json

import numpy as np

from utils import normalize_rows


class NN:
    def __init__(self, vectors_path, vocab_path, normalize=True):

        self.vectors_path = vectors_path
        self.vocab_path = vocab_path
        self.normalize = normalize

        self.load_vocab()
        self.load_vectors()

    def nn_search(self, word):
        pass

    def load_vectors(self):
        self.db = np.load(self.vectors_path, allow_pickle=True)
        if self.normalize:
            self.db = normalize_rows(self.db)
        self.dim = self.db.shape[1]

    def load_vocab(self):
        self.id2token = json.load(open(self.vocab_path, 'r'))
        self.id2token = {int(i): w for i, w in self.id2token.items()}
        self.token2id = {w: i for i, w in self.id2token.items()}

    def get_query_vec(self, token):
        ix = self.token2id[token]
        return self.db[ix].reshape(1, -1)

    def rand_query(self):
        return np.random.choice(list(self.token2id.keys()))

    def rand_neighbors(self, query, n, thresh):
        assert n >= 1 and n < 100, "n must be in range(1,100)"

        most_similars, scores = self.nn_search(query, 201)
        neighbors = []
        intruder = most_similars[-1]
        for token, score in zip(most_similars[:10], scores[:10]):
            if token != query and score > thresh:
                neighbors.append(token)
        if n <= len(neighbors):
            neighbors = np.random.choice(neighbors, size=n, replace=False)
            return list(neighbors), intruder
        return [], None


class NaiveNN(NN):
    def nn_search(self, q_token, top_k):
        cosine = lambda a, db: np.dot(a, self.db.T)
        vec = self.get_query_vec(q_token)
        scores = cosine(vec, self.db)[0]
        indecies = scores.argsort()[-topk:][::-1]
        D = [self.id2token[i] for i in indecies]
        I = [scores[i] for i in indecies]
        I = np.squeeze(I)
        D = np.squeeze(D)
        return D, I


class AproximateNN(NN):
    def load_vectors(self):
        self.db = np.load(self.vectors_path, allow_pickle=True)
        if self.normalize:
            self.db = normalize_rows(self.db)
        self.dim = self.db.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.db = np.ascontiguousarray(self.db, dtype=np.float32)
        self.index.add(self.db)

    def nn_search(self, q_token, topk):
        vec = self.get_query_vec(q_token)
        I, D = self.index.search(vec.reshape(1, -1), topk)
        I = np.squeeze(I)
        D = np.squeeze(D)
        D = [self.id2token[i] for i in D]
        return D, I


nn = NaiveNN
try:
    import faiss

    nn = AproximateNN
except:
    print("Warning: Faiss not supported")
