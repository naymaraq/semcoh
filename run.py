import argparse

from surveys import CoherenceSurvey, RelatednessSurvey
from utils import print_ascii

ARG = argparse.ArgumentParser()

ARG.add_argument('--task', type=str, help='coh or rel')
ARG.add_argument('--q', type=str, default=None, help='Path to query list')
ARG.add_argument('--pairs', type=str, default=None, help='Path to pairs list')
ARG.add_argument('--v', type=str, help='Path to vectors.npy')
ARG.add_argument('--i', type=str, help='Path to index.json')
ARG.add_argument('--f', type=str, default=None, help='Path to freqs.json')
ARG.add_argument('--n', type=bool, default=True, help='Normalize')
ARG = ARG.parse_args()

if ARG.task == 'coh':
    print_ascii("Coherence of Semantic Space")
    survey = CoherenceSurvey(ARG.v, ARG.i, ARG.f, ARG.q, ARG.n)
elif ARG.task == 'rel':
    print_ascii("Relatedness of Semantic Space")
    survey = RelatednessSurvey(ARG.q, ARG.pairs)

survey.start_survey()
