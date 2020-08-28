import argparse

from survey import Survey
from utils import print_ascii

ARG = argparse.ArgumentParser()
ARG.add_argument('--v', type=str, help='Path to vectors.npy')
ARG.add_argument('--i', type=str, help='Path to index.json')
ARG.add_argument('--f', type=str, default=None, help='Path to freqs.json')
ARG.add_argument('--q', type=str, default=None, help='Patt to query list')
ARG.add_argument('--n', type=bool, default=True, help='Normalize')

ARG = ARG.parse_args()

print_ascii("Coherence of Semantic Space")
survey = Survey(ARG.v, ARG.i, ARG.f, ARG.q, ARG.n)
survey.start_survey()
