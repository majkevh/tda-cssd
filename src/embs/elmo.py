import numpy as np
from smart_open import open
from simple_elmo import ElmoModel
import logging
import time
from pathlib import Path
from helpers import dict_languages
# Global hyperparameter
BATCH_SIZE = 196
WORD_LIMIT = 512
CACHE_SIZE = 12800
LAYER = "top"

class ContextEmbedderELMo:
    def __init__(self, language, corpus):
        """
        Initialize the ContextEmbedderELMo class with required paths and constants.
        """
        
        # Define paths
        base_dir = Path(__file__).resolve().parents[2]  
        self.data_path = base_dir / 'data' / language / corpus / "lemma" / dict_languages[language][corpus]
        self.vocab_path = base_dir / 'data' / language / "targets.txt"
        self.model_path = dict_languages[language]["ELMo-model"]
        self.output_file = base_dir / 'data' / language / "embeddings" / "ELMo" / f"{corpus}.npz"
        
        # Initialize variables
        self.embeddings = {}
        self.model = ElmoModel()

        # Logger setup
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    def _load_vocab_and_count(self):
        """Load vocabulary file and count occurrences in the dataset."""

        # Load vocabulary
        with open(self.vocab_path, "r") as f:
            self.embeddings = {line.strip(): 0 for line in f}

        # Count occurrences
        wordcount = 0
        with open(self.data_path, "r") as corpus:
            for line in corpus:
                res = line.strip().split()[:WORD_LIMIT]
                for word in res:
                    if word in self.embeddings:
                        self.embeddings[word] += 1
                        wordcount += 1

        self.logger.info(f"Occurrences of target words: {wordcount}")
        for word, count in self.embeddings.items():
            self.logger.info(f"Word: {word}, Occurrences: {count}")

    def _initialize_model(self):
        """Initialize the model and vector placeholders."""

        # Load pre-trained ELMo model
        self.model.load(self.model_path, max_batch_size=BATCH_SIZE)
        
        # Prepare vectors and word counts
        self.embeddings = {
            word: np.zeros((int(self.embeddings[word]), self.model.vector_size))
            for word in self.embeddings
        }
        self.counts = {w: 0 for w in self.embeddings}


    def _update_vectors(self, lines_cache):
        """Update vector embeddings for cached lines."""
        elmo_vectors = self.model.get_elmo_vectors(lines_cache, layers=LAYER)
        for sent, matrix in zip(lines_cache, elmo_vectors):
            for word, vector in zip(sent, matrix):
                if word in self.embeddings:
                    self.embeddings[word][self.counts[word], :] = vector
                    self.counts[word] += 1


    def run(self):
        self.logger.info("Starting ELMo embedding processing...")
        self._load_vocab_and_count()
        self._initialize_model()
        
        start_time = time.time()
        lines_cache = []
        lines_processed = 0

        # Open the dataset and process lines
        with open(self.data_path, "r") as dataset:
            for line in dataset:
                # Split the line and check if any target words are in the vocabulary
                words_in_line = line.strip().split()[:WORD_LIMIT]
                if any(word in self.embeddings for word in words_in_line):
                    lines_cache.append(words_in_line)
                    lines_processed += 1

                # If cache size limit is reached, process the cache and clear it
                if len(lines_cache) >= CACHE_SIZE:
                    self._update_vectors(lines_cache)
                    lines_cache.clear()  
                    self.logger.info(f"Lines processed: {lines_processed}")

        # Process any remaining lines in the cache
        if lines_cache:
            self._update_vectors(lines_cache)

        # Log the total processing time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Processing completed in {elapsed_time:.2f} seconds.")

        self.embeddings = {
            w.split('_', 1)[0]: self.embeddings[w][~(self.embeddings[w] == 0).all(1)]
            for w in self.embeddings
        }
        np.savez_compressed(self.output_file, **self.embeddings)