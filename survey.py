import argparse
from nearest_neighbors import nn

class Survey:

    def __init__(self, vectors_path, vocab_path, freqs, normalize):
        self.nn_obj = nn(vectors_path, vocab_path, normalize=normalize)

    def start_survey(self):
        pass



ARG = argparse.ArgumentParser()
ARG.add_argument('--v', type=str, help='Path to vectors.npy')
ARG.add_argument('--i', type=str, help='Path to index.json')
ARG.add_argument('--f', type=str, default='', help='Path to freqs.json')
ARG.add_argument('--n', type=bool, default=True, help='Normalize')

ARG = ARG.parse_args()


Survey(ARG.v, ARG.i, ARG.f, ARG.n)




