import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from node2vec import Node2Vec
from typing import List






































def get_gold(word: str, language: str) -> List[float]:
    """
    Retrieve gold standard scores for a given word in a specific language.

    Parameters:
    - word: The target word to retrieve gold scores for.
    - language: The language to retrieve the gold scores from.

    Returns:
    - A list of gold scores (float) from both graded and binary truth files.
    """
    gold_scores = []
    try:
        with open(f'./data/{language}/truth/graded.txt', 'r') as graded_file:
            for line in graded_file:
                parts = line.strip().split()
                if parts[0].split('_')[0] == word:
                    gold_scores.append(float(parts[1]))

        with open(f'./data/{language}/truth/binary.txt', 'r') as binary_file:
            for line in binary_file:
                parts = line.strip().split()
                if parts[0].split('_')[0] == word:
                    gold_scores.append(float(parts[1]))
                    break  # Stop after finding the first binary score
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check if the files exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return gold_scores

def jaccard_distance(G1: nx.Graph, G2: nx.Graph) -> float:
    """
    Compute the Jaccard distance between two graphs based on their edge sets.

    Parameters:
    - G1: First input graph.
    - G2: Second input graph.

    Returns:
    - Jaccard distance (float) between the edge sets of G1 and G2.
    """
    edges_G1, edges_G2 = set(G1.edges()), set(G2.edges())
    intersection_size = len(edges_G1.intersection(edges_G2))
    union_size = len(edges_G1.union(edges_G2))
    return 1 - intersection_size / union_size if union_size != 0 else 0.0

def degree_distribution_distance(G1: nx.Graph, G2: nx.Graph) -> float:
    """
    Compute the Wasserstein distance between the degree distributions of two graphs.

    Parameters:
    - G1: First input graph.
    - G2: Second input graph.

    Returns:
    - Wasserstein distance (float) between the degree distributions of G1 and G2.
    """
    degrees_G1 = np.array([degree for _, degree in G1.degree()])
    degrees_G2 = np.array([degree for _, degree in G2.degree()])
    return wasserstein_distance(degrees_G1, degrees_G2)

def graph_embedding_node2vec(G: nx.Graph, num_walks: int = 100, workers: int = 4) -> np.ndarray:
    """
    Compute the Node2Vec embedding of a graph and return the average node embeddings.

    Parameters:
    - G: Input graph to be embedded.
    - num_walks: Number of random walks per node (default=100).
    - workers: Number of CPU cores to use for parallel processing (default=4).

    Returns:
    - A numpy array representing the average node embedding of the graph.
    """
    node2vec = Node2Vec(G, num_walks=num_walks, workers=workers, quiet=True)
    model = node2vec.fit()
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return np.mean(node_embeddings, axis=0)

def random_walk_distance(G1: nx.Graph, G2: nx.Graph) -> float:
    """
    Compute the distance between two graphs based on their Node2Vec embeddings.

    Parameters:
    - G1: First input graph.
    - G2: Second input graph.

    Returns:
    - L2 (Euclidean) distance (float) between the Node2Vec embeddings of G1 and G2.
    """
    embedding_G1 = graph_embedding_node2vec(G1)
    embedding_G2 = graph_embedding_node2vec(G2)
    return np.linalg.norm(embedding_G1 - embedding_G2)

def von_neumann_entropy(G: nx.Graph) -> float:
    """
    Compute the von Neumann entropy of a graph based on its Laplacian matrix.

    Parameters:
    - G: Input graph.

    Returns:
    - Von Neumann entropy (float) of the graph.
    """
    L = nx.laplacian_matrix(G).todense()
    eigenvalues = np.linalg.eigvals(L)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Ignore zero or negative eigenvalues
    return -np.sum(eigenvalues * np.log(eigenvalues))

def von_neumann_entropy_distance(G1: nx.Graph, G2: nx.Graph) -> float:
    """
    Compute the von Neumann entropy distance between two graphs.

    Parameters:
    - G1: First input graph.
    - G2: Second input graph.

    Returns:
    - Absolute difference (float) in von Neumann entropy between G1 and G2.
    """
    entropy_G1 = von_neumann_entropy(G1)
    entropy_G2 = von_neumann_entropy(G2)
    return abs(entropy_G1 - entropy_G2)