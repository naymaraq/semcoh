import argparse

ARG = argparse.ArgumentParser()
ARG.add_argument('--v', type=str, help='Path to vectors.npy')
ARG.add_argument('--i', type=str, help='Path to index.json')
ARG.add_argument('--f', type=str, help='Path to freqs.json')
ARG = ARG.parse_args()


