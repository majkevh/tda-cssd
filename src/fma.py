import argparse
import numpy as np
from pathlib import Path
import os
import logging
import time
from scipy.spatial.distance import squareform, pdist
from gudhi.wasserstein.barycenter import lagrangian_barycenter as bary
from tqdm import tqdm
import sys

# Adjust the path for correct module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.kcluster as kH0
from ripser import ripser

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def compute_subsampling(embeddings_C1P, embeddings_C2N, args, N_OBJ, n_samples):
    """
    Perform subsampling and compute persistence or k-clustering diagrams.

    Parameters:
    - embeddings_C1P: Embeddings for corpus 1.
    - embeddings_C2N: Embeddings for corpus 2.
    - args: Parsed arguments (including metric, k-value, etc.).
    - N_OBJ: Number of subsampling iterations.
    - n_samples: Number of samples for each subsampling.

    Returns:
    - dgms_C1P: Persistence diagrams for corpus 1.
    - dgms_C2N: Persistence diagrams for corpus 2.
    """
    # Initialize diagram dictionaries, discarding H0_0 if args.k == -1 (persistent homology case)
    homology_keys = [f"H0_{args.k}"] if args.k != -1 else ["H1", "H2"]
    dgms_C1P, dgms_C2N = {key: [] for key in homology_keys}, {key: [] for key in homology_keys}

    # Subsample embeddings and compute diagrams
    for _ in tqdm(range(N_OBJ), desc="Subsampling"):
        indices_C1P = np.random.choice(embeddings_C1P.shape[0], n_samples, replace=False)
        indices_C2N = np.random.choice(embeddings_C2N.shape[0], n_samples, replace=False)

        graph_C1P = squareform(pdist(embeddings_C1P[indices_C1P], metric=args.gmetric))
        graph_C2N = squareform(pdist(embeddings_C2N[indices_C2N], metric=args.gmetric))

        if args.k != -1:
            # Compute k-clustering diagrams
            dgm1 = kH0.computeDiagram(graph_C1P, k=args.k)
            dgm2 = kH0.computeDiagram(graph_C2N, k=args.k)
            dgms_C1P[f"H0_{args.k}"].append(dgm1[np.isfinite(dgm1[:, 1])])
            dgms_C2N[f"H0_{args.k}"].append(dgm2[np.isfinite(dgm2[:, 1])])
        else:
            # Compute persistent homology diagrams, skipping H0_0
            # Only compute H1 and H2 (loops and voids)
            for idx, (dgm1, dgm2) in enumerate(zip(ripser(graph_C1P, distance_matrix=True, maxdim=2)['dgms'][1:],
                                                   ripser(graph_C2N, distance_matrix=True, maxdim=2)['dgms'][1:])):
                key = homology_keys[idx]
                dgms_C1P[key].append(dgm1[np.isfinite(dgm1[:, 1])])
                dgms_C2N[key].append(dgm2[np.isfinite(dgm2[:, 1])])

    return dgms_C1P, dgms_C2N

def main():
    # Start timing the process
    start_time = time.time()

    # Argument parser setup
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--language", "-l", help="Choose language to use", required=True, choices=["english", "german", "latin", "swedish"])
    arg("--embedding", "-e", help="Choose contextualized embedding to compute topological summary on", required=True, choices=["BERT", "ELMo", "XLM-R"])
    arg("--verbose", "-v", help="Set verbose parameter", default=True, required=False, choices=[True, False])
    arg("--k", "-k", help="Choose between ripser PH (-1) or k-clustering (specify k value)", required=True, type=int)
    arg("--gmetric", "-gm", help="Choose metric to compute graph adjacency matrix, suggested either euclidean or cosine distance", default="cosine", required=False, choices=['cosine', 'euclidean'])
    args = parser.parse_args()

    # Constants
    LIMIT_POOL = 1000
    N_OBJ = 20
    P = 0.8

    # Define paths
    data_path = Path(f"./data/{args.language}/")
    targets_path = data_path / "targets.txt"
    main_folder_path = Path(f"./tests/FMa/{args.embedding}/")
    main_folder_path.mkdir(parents=True, exist_ok=True)
    embeddings_dir = data_path / "embeddings"

    # Load target words
    with targets_path.open('r', encoding='utf-8') as f_in:
        targets = [line.split('_', 1)[0].strip() for line in f_in]

    # Load embedding data
    data_c1p = np.load(embeddings_dir / f"{args.embedding}/corpus1.npz")
    data_c2n = np.load(embeddings_dir / f"{args.embedding}/corpus2.npz")

    for word in targets:
        # Retrieve embeddings for the target word
        embeddings_C1P = data_c1p[word]
        embeddings_C2N = data_c2n[word]

        if args.verbose:
            logger.info(f"Word: {word} - Dataset1: {embeddings_C1P.shape[0]} sentences, Dataset2: {embeddings_C2N.shape[0]} sentences")

        # Minimum sample size for the two corpora
        n_min = min(embeddings_C1P.shape[0], embeddings_C2N.shape[0], LIMIT_POOL)
        n_samples = int(P * n_min)

        # Compute subsampling and persistence diagrams
        dgms_C1P, dgms_C2N = compute_subsampling(embeddings_C1P, embeddings_C2N, args, N_OBJ, n_samples)

        # Create subfolder for saving results
        subfolder_path = main_folder_path / args.language / word
        subfolder_path.mkdir(parents=True, exist_ok=True)

        # Compute and save weighted means using barycenter
        for homology in tqdm(dgms_C1P.keys(), desc="Computing weighted means"):
            wmean_C1P = bary(dgms_C1P[homology], init=0, verbose=False)
            wmean_C2N = bary(dgms_C2N[homology], init=0, verbose=False)

            # Save results
            np.save(os.path.join(subfolder_path, f"{word}_c1p_{homology}.npy"), wmean_C1P)
            np.save(os.path.join(subfolder_path, f"{word}_c2n_{homology}.npy"), wmean_C2N)

    # End timing and log the total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()