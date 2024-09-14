
import os
import warnings
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer, BertModel
from helpers import  OneTokenContext, dict_languages
from pathlib import Path
from gensim.models.word2vec import LineSentence

LAYER = "top"
CONTEXT_SIZE= 64
BATCH_SIZE = 64
DIM = 768 #all the pre-trained considered models have same embedding dimensionality

class ContextEmbedderBERT:
    def __init__(self, language, corpus):
        """
        Initialize the ContextEmbedderBERT class with required paths and constants.
        """

        # Define paths
        base_dir = Path(__file__).resolve().parents[2] 
        self.data_path = base_dir / 'data' / language / corpus / "lemma" / dict_languages[language][corpus]
        self.vocab_path = base_dir / 'data' / language / "targets.txt"
        self.model_name = dict_languages[language]["BERT-model"]
        self.output_file = base_dir / 'data' / language / "embeddings" / "BERT" / f"{corpus}.npz"

        # Logger setup
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    def set_seed(self, seed):
        """
        Set random seed.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def _initialize_model(self):
        """
        Initialize BERT model and Tokenizer.
        """
        with self.vocab_path.open('r', encoding='utf-8') as f_in:
            self.targets = [line.split('_', 1)[0].strip() for line in f_in]

        self.logger.info(f"Target words: {self.targets}")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, never_split=self.targets)
        self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.model.to(self.device)

    def _add_tokens_to_vocabulary(self):
        """
        Adds target tokens to the tokenizer if they are not already present and adjusts model embeddings.
        """
        unk_id = self.tokenizer.convert_tokens_to_ids('[UNK]')
        self.i2w = {}
        
        targets_ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in self.targets]
        assert len(self.targets) == len(targets_ids), "Mismatch between targets and encoded target IDs"

        for token, token_ids in zip(self.targets, targets_ids):
            if len(token_ids) > 1 or (token_ids == [unk_id]):
                if self.tokenizer.add_tokens([token]):
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    self.i2w[len(self.tokenizer) - 1] = token
                else:
                    self.logger.error(f"Failed to add token '{token}' to tokenizer: {self.tokenizer.tokenize(token)}")
            else:
                self.i2w[token_ids[0]] = token
        
    def collect_usages(self, sentences):
        """
        Collect BERT representations for target words in sentences and count their occurrences.
        """
        warnings.filterwarnings("ignore")  # Suppress all warnings during this block
        
        target_counter = {target: 0 for target in self.i2w}
        
        # Count occurrences of target words
        self.n_sentences = 0
        for sentence in sentences:
            self.n_sentences += 1
            encoded_sentence = self.tokenizer.encode(' '.join(sentence), add_special_tokens=False)
            for tok_id in encoded_sentence:
                if tok_id in target_counter:
                    target_counter[tok_id] += 1

        self.logger.info(f"Usages found: {sum(target_counter.values())}")
        for target, count in target_counter.items():
            self.logger.info(f'{self.i2w[target]}: {count}')

        # Initialize usage matrices and index tracker
        N_LAYERS = 1 if LAYER in ["top", "average"] else 13 # 13 is the standard 12 layer BERT + encoder layer
        self.embeddings = {self.i2w[t]: np.empty((count, N_LAYERS * DIM)) for t, count in target_counter.items()}
        self.curr_idx = {self.i2w[t]: 0 for t in target_counter}

    def _update_embeddings(self, batch_input_ids, batch_lemmas, batch_target_pos, hidden_states):
        """
        Updates the embeddings with the hidden states from the model for each word's usage.
        """
        # Iterate over each item in the batch
        for b_id in np.arange(len(batch_input_ids)):
            lemma = batch_lemmas[b_id]

            # Extract hidden states for the given word's position in each layer
            layers = [layer[b_id, batch_target_pos[b_id] + 1, :] for layer in hidden_states]

            if LAYER == "top":
                emb_vector = layers[-1]
            elif LAYER == "all":
                emb_vector = np.concatenate(layers)
            elif LAYER == "average":
                emb_vector = np.mean(layers, axis=0)
            else:
                raise ValueError(f"Invalid method for extracting embeddings. Choose 'top', 'average', or 'all'.")

            # Store the usage vector in the corresponding lemma's usage matrix
            self.embeddings[lemma][self.curr_idx[lemma], :] = emb_vector

            # Update the index tracker for the lemma
            self.curr_idx[lemma] += 1

    def _process_batches(self, dataloader):
        """
        Iterates over batches of data, passes them through the model, and collects embeddings.
        """
        for stp, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            self.model.eval()

            # Move the batch data to the correct device
            batch_tuple = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)

            # Extract the relevant inputs from the batch
            batch_input_ids = batch_tuple[0].squeeze(1)  # Token IDs
            batch_lemmas = batch_tuple[1]  # Lemmas (target words)
            batch_target_pos = batch_tuple[2]    # Token positions for target words

            with torch.no_grad():
                if torch.cuda.is_available():
                    batch_input_ids = batch_input_ids.to('cuda')

                # Forward pass through the model
                outputs = self.model(batch_input_ids)

                hidden_states = [layer.detach().cpu().numpy() if torch.cuda.is_available() else layer.numpy() for layer in outputs.hidden_states]

                # Update embeddings with the hidden states for this batch
                self._update_embeddings(batch_input_ids, batch_lemmas, batch_target_pos, hidden_states)



    def run(self):
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.logger.info(f"Using device: {self.device}")

        self.set_seed(42)
        self._initialize_model()

        self._add_tokens_to_vocabulary()

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Load the corpus
        sentences = LineSentence(self.data_path)
        self.collect_usages(sentences)
        dataset = OneTokenContext(self.i2w, sentences, CONTEXT_SIZE, self.tokenizer, self.n_sentences)

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

       # Process the batches
        self._process_batches(dataloader)

        # Save file
        np.savez_compressed(self.output_file, **self.embeddings)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Processing completed in {elapsed_time:.2f} seconds.")



































































