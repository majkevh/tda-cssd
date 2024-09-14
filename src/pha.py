import argparse
import numpy as np
from pathlib import Path
import os
import logging
import time
from scipy.spatial.distance import squareform, pdist
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.kcluster as kH0
from ripser import ripser

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up logging similar to the attached file
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
    arg("--limsamples", "-ls", help="Same sample size between C1 and C2", required=False, default=True, choices=[True, False])
    arg("--gmetric", "-gm", help="Choose metric to compute graph adjacency matrix, suggested either euclidean or cosine distance", default="cosine", required=False, choices=['cosine', 'euclidean'])
    args = parser.parse_args()

    # Constants
    LIMIT_POOL = 1000

    # Define paths
    data_path = Path(f"./data/{args.language}/")
    targets_path = data_path / "targets.txt"
    main_folder_path = Path(f"./tests/PHa/{args.embedding}/")
    main_folder_path.mkdir(parents=True, exist_ok=True)
    embeddings_dir = data_path / "embeddings"

    with targets_path.open('r', encoding='utf-8') as f_in:
            targets = [line.split('_', 1)[0].strip() for line in f_in]

    # Load embedding data
    data_c1p = np.load(embeddings_dir / f"{args.embedding}/corpus1.npz")
    data_c2n = np.load(embeddings_dir / f"{args.embedding}/corpus2.npz")

    for word in targets:
        embeddings_C1P = data_c1p[word]
        embeddings_C2N = data_c2n[word]

        if args.verbose:
            logger.info(f"Word: {word} - Dataset1: {embeddings_C1P.shape[0]} sentences, Dataset2: {embeddings_C2N.shape[0]} sentences")

        # Limit pool of sampling if needed
        if args.limsamples:
            n_samples = min(embeddings_C1P.shape[0], embeddings_C2N.shape[0], LIMIT_POOL)
            embeddings_C1P = embeddings_C1P[np.random.choice(embeddings_C1P.shape[0], n_samples, replace=False)]
            embeddings_C2N = embeddings_C2N[np.random.choice(embeddings_C2N.shape[0], n_samples, replace=False)]

        # Calculate distance matrices
        graph_C1P = squareform(pdist(embeddings_C1P, metric=args.gmetric))
        graph_C2N = squareform(pdist(embeddings_C2N, metric=args.gmetric))

        # Create subfolder for saving results
        subfolder_path = main_folder_path / args.language / word
        subfolder_path.mkdir(parents=True, exist_ok=True)

        # Use tqdm to show progress
        if args.k != -1:
            dmgs_C1P = kH0.computeDiagram(graph_C1P, k=args.k)
            dmgs_C2N = kH0.computeDiagram(graph_C2N, k=args.k)

            # Filter out infinite values
            dmgs_C1P = dmgs_C1P[np.isfinite(dmgs_C1P[:, 1])]
            dmgs_C2N = dmgs_C2N[np.isfinite(dmgs_C2N[:, 1])]

            # Save results
            np.save(subfolder_path / f"{word}_c1p_H0_{args.k}.npy", dmgs_C1P)
            np.save(subfolder_path / f"{word}_c2n_H0_{args.k}.npy", dmgs_C2N)
        else:
            # Compute persistent homology diagrams for both graphs
            homology_keys = ["H0_0", "H1", "H2"]
            for idx, (dgm1, dgm2) in enumerate(zip(ripser(graph_C1P, distance_matrix=True, maxdim=2)['dgms'],
                                                   ripser(graph_C2N, distance_matrix=True, maxdim=2)['dgms'])):
                key = homology_keys[idx]
                # Filter out infinite values
                dgm1_filtered = dgm1[np.isfinite(dgm1[:, 1])]
                dgm2_filtered = dgm2[np.isfinite(dgm2[:, 1])]

                # Save results
                np.save(subfolder_path / f"{word}_c1p_{key}.npy", dgm1_filtered)
                np.save(subfolder_path / f"{word}_c2n_{key}.npy", dgm2_filtered)

    # End timing and log the total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()