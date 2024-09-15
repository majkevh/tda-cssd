import argparse
import numpy as np
from pathlib import Path
import logging
import time
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle
import sys
from persim import bottleneck, sliced_wasserstein, PersistenceImager
from gudhi.representations.vector_methods import BettiCurve
from gudhi.representations import Landscape
from gudhi.wasserstein import wasserstein_distance
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_distances
import os
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.aux_functions as aux

# Suppress warnings and configure logging
warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def MRa_distances(args, subfolder_path, word):
    """Calculate MRa distances based on the selected metric."""
    g1 = pickle.load(open(subfolder_path / f"graphC1P_{args.dreduction}.pkl", 'rb'))
    g2 = pickle.load(open(subfolder_path / f"graphC2N_{args.dreduction}.pkl", 'rb'))
    metric_funcs = {
        "jaccard": aux.jaccard_distance,
        "degree-distr": aux.degree_distribution_distance,
        "rw": aux.random_walk_distance,
        "entropy": aux.von_neumann_entropy_distance
    }
    return metric_funcs[args.metric](g1, g2)

def Rips_distances(args, subfolder_path, word):
    """Calculate persistence diagram based distances using persistence diagrams."""
    word1_path = f"{word}_c1p_H{abs(args.k)}.npy" if args.k < 0 else f"{word}_c1p_H0_{args.k}.npy"
    word2_path = f"{word}_c2n_H{abs(args.k)}.npy" if args.k < 0 else f"{word}_c2n_H0_{args.k}.npy"
    dgm1 = np.load(subfolder_path /word1_path)
    dgm2 = np.load(subfolder_path / word2_path)

    metric_funcs = {
        "bottleneck": lambda: bottleneck(dgm1, dgm2),
        "wasserstein": lambda: wasserstein_distance(dgm1, dgm2, order=2, internal_p=2),
        "sliced-ws": lambda: sliced_wasserstein(dgm1, dgm2),
        "pers-images": lambda: np.linalg.norm(PersistenceImager().transform(dgm1) - PersistenceImager().transform(dgm2)),
        "betti-curve": lambda: np.linalg.norm(BettiCurve(resolution=100).fit_transform([dgm1])[0] - BettiCurve(resolution=100).fit_transform([dgm2])[0]),
        "pers-landscape": lambda: np.linalg.norm(Landscape(resolution=100).fit_transform([dgm1]).flatten() - Landscape(resolution=100).fit_transform([dgm2]).flatten()),
        "u-mean": lambda: np.abs(np.mean(np.log(np.log(dgm1[:, 1] / dgm1[:, 0]))) - np.mean(np.log(np.log(dgm2[:, 1] / dgm2[:, 0])))),
        "u-ks": lambda: ks_2samp(np.log(np.log(dgm1[:, 1] / dgm1[:, 0])), np.log(np.log(dgm2[:, 1] / dgm2[:, 0]))).statistic
    }
    return metric_funcs[args.metric]()

def main():
    # Start timing the process
    start_time = time.time()

    
    # Argument parser setup
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--language", "-l", help="Choose language to use", required=True, choices=["english", "german", "latin", "swedish"])
    arg("--embedding", "-e", help="Choose contextualized embedding to compute topological summary on", required=True, choices=["BERT", "ELMo", "XLM-R"])
    arg("--verbose", "-v", help="Set verbose parameter", default=True, required=False, choices=[True, False])
    arg("--k", "-k", help="Choose between ripser PH (choose 0 for H0, -1 for H1, -2 for H2) or k-clustering (specify k value >1)", required=False, type=int)
    arg("--ensamble", "-ens", help="Choose non-topological method to ensamble with or 'pure' to use only topological distances", required=True, choices=['apd', "cd", 'pure', "cmb"])
    arg("--algorithm", "-a", help="Choose topological SSD algorithm between FMa, PHa, MRa", required=True, choices=["FMa", "PHa", "MRa"])
    arg("--dreduction", "-dr", help="Choose dimensionality reduction for MRa algorithm", required=False, default="Isomap", choices=["UMAP", "PCA", "TSNE", "Isomap"])
    arg("--metric", "-m", help="Choose metric for assessing topological distance", required=True,  choices=["bottleneck", "wasserstein", "sliced-ws", "pers-images", "betti-curve", "pers-landscape", "u-mean", "u-ks", "jaccard", "rw", "degree-distr", "entropy"])
    arg("--output_path", "-o", help="Specify output path for .txt file with predictions", default= "./output.txt", required=False, type=str)
    arg("--input_path", "-i", help="Path to main simulation folder", required=False, default="./precomputed/", type=str)
    args = parser.parse_args()
    assert args.metric in (["jaccard", "rw", "degree-distr", "entropy"] if args.algorithm == "MRa" else ["bottleneck", "wasserstein", "sliced-ws", "pers-images", "betti-curve", "pers-landscape", "u-mean", "u-ks"])


    data_path = Path(f"./data/{args.language}/")
    main_folder_path = Path(f"{args.input_path}/{args.algorithm}/{args.embedding}/")
    embeddings_dir = data_path / "embeddings"

    targets = [line.split('_', 1)[0].strip() for line in data_path.joinpath("targets.txt").open('r', encoding='utf-8')]
    data_c1p = np.load(embeddings_dir / f"{args.embedding}/corpus1.npz")
    data_c2n = np.load(embeddings_dir / f"{args.embedding}/corpus2.npz")

    predictions, bias_apd, bias_cd = [], [], []

    for word in tqdm(targets, desc="Processing words"):
        embeddings_C1P, embeddings_C2N = data_c1p[word], data_c2n[word]
        subfolder_path = main_folder_path / args.language / word

        if args.algorithm == "MRa":
            dist = MRa_distances(args, subfolder_path, word)
        else:
            dist = Rips_distances(args, subfolder_path, word)

        predictions.append(dist)

        if args.ensamble != "pure":
            bias_cd.append(cosine(np.mean(embeddings_C1P, axis=0), np.mean(embeddings_C2N, axis=0)))
            PD = cosine_distances(embeddings_C1P, embeddings_C2N)
            bias_apd.append(np.mean(np.mean(PD, axis=1)))

    if args.ensamble == "pure":
        np.savetxt(args.output_path, predictions, fmt='%f')
    else:
        scaler = MinMaxScaler()
        bias_cd_std = scaler.fit_transform(np.array(bias_cd).reshape(-1, 1))
        bias_apd_std = scaler.fit_transform(np.array(bias_apd).reshape(-1, 1))
        topo_std = scaler.fit_transform(np.array(predictions).reshape(-1, 1))

        bias_method = bias_cd_std if args.ensamble == "cd" else bias_apd_std if args.ensamble == "apd" else (bias_cd_std * bias_apd_std).reshape(-1, 1)
        features = np.sqrt(topo_std*bias_method)
        np.savetxt(args.output_path, features, fmt='%f')

    logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()