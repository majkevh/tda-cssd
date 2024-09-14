from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE, Isomap
import argparse
import kmapper as km
import pickle
import numpy as np
from pathlib import Path
import os
import logging
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    arg("--limsamples", "-ls", help="Same sample size between C1 and C2", required=False, default=True, choices=[True, False])
    arg("--pjmetric", "-pm", help="Choose metric for projection of data (for example in TSNE or UMAP projection), suggested either euclidean or cosine distance", default="euclidean", required=False, choices=['cosine', 'euclidean'])
    args = parser.parse_args()

    # Constants
    LIMIT_POOL = 2000
    N_CUBES = 10
    P_OVERLAP = 0.5

    DB_METRIC = "cosine"
    DB_EPS = 10
    DB_MIN_SAMPLES = 2

    # Define paths
    data_path = Path(f"./data/{args.language}/")
    targets_path = data_path / "targets.txt"
    main_folder_path = Path(f"./tests/MRa/{args.embedding}/")
    main_folder_path.mkdir(parents=True, exist_ok=True)
    embeddings_dir = data_path / "embeddings"

    with targets_path.open('r', encoding='utf-8') as f_in:
            targets = [line.split('_', 1)[0].strip() for line in f_in]

    # Load embedding data
    data_c1p = np.load(embeddings_dir / f"{args.embedding}/corpus1.npz")
    data_c2n = np.load(embeddings_dir / f"{args.embedding}/corpus2.npz")


    mapper = km.KeplerMapper(verbose=-1)
    cover = km.Cover(n_cubes=N_CUBES, perc_overlap=P_OVERLAP)


    projections = [
        TSNE(n_components=2, metric=args.pjmetric),
        Isomap(n_components=2, metric=args.pjmetric),
        PCA(n_components=2),
        UMAP(n_components=2, metric=args.pjmetric)
    ]
    
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

        # Create subfolder for saving results
        subfolder_path = main_folder_path / args.language / word
        subfolder_path.mkdir(parents=True, exist_ok=True)

        for projection_method in projections:
            # project data into 2D subspace
            projection_name = projection_method.__class__.__name__

            projected_dataC1P = mapper.fit_transform(embeddings_C1P, projection=[projection_method])
            projected_dataC2N = mapper.fit_transform(embeddings_C2N, projection=[projection_method])

            # cluster data using DBSCAN and create the mapper graph with finer cover
            G1P = mapper.map(projected_dataC1P, embeddings_C1P, clusterer=DBSCAN(metric=DB_METRIC, eps=DB_EPS, min_samples=DB_MIN_SAMPLES), cover=cover)
            G2N = mapper.map(projected_dataC2N, embeddings_C2N, clusterer=DBSCAN(metric=DB_METRIC, eps=DB_EPS, min_samples=DB_MIN_SAMPLES), cover=cover)

            G1P_NX = km.adapter.to_nx(G1P)
            G2N_NX = km.adapter.to_nx(G2N)

            with open(subfolder_path / f"graphC1P_{projection_name}.pkl", 'wb') as f:
                pickle.dump(G1P_NX, f)
                
            with open(subfolder_path / f"graphC2N_{projection_name}.pkl", 'wb') as f:
                pickle.dump(G2N_NX, f)

    # End timing and log the total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
