import os
import warnings
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForMaskedLM, AutoTokenizer
from helpers import  MultipleTokenContext, dict_languages
from pathlib import Path
from collections import defaultdict
from gensim.models.word2vec import LineSentence

LAYER = "average"
CONTEXT_SIZE= 512
BATCH_SIZE = 32
DIM = 768 # pre-trained model dimensionality

class ContextEmbedderXLMR:
    def __init__(self, language, corpus):
        """
        Initialize the ContextEmbedderXLMR class with required paths and constants.
        """

        # Define paths
        base_dir = Path(__file__).resolve().parents[2] 
        self.data_path = base_dir / 'data' / language / corpus / "lemma" / dict_languages[language][corpus]
        self.vocab_path = base_dir / 'data' / language / "targets.txt"
        self.model_name = dict_languages[language]["XLMR-model"]
        self.output_file = base_dir / 'data' / language / "embeddings" / "XLM-R" / f"{corpus}.npz"

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
        Initialize RoBERTa model and Tokenizer.
        """
        with self.vocab_path.open('r', encoding='utf-8') as f_in:
            self.targets = [line.strip() for line in f_in]

        self.logger.info(f"Target words: {self.targets}")

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base", output_hidden_states=True)
        self.model.to(self.device)

    def _store_vocabulary(self):
        """
        Stores vocabulary indexes for the target words list in a more efficient and concise manner.
        """
        targets_ids = {
            word: self.tokenizer.encode(word, add_special_tokens=False) 
            for word in self.targets
        }

        ids2lemma = {}
        lemma2ids = defaultdict(list)
        max_token_length = 0

        for lemma, token_ids in targets_ids.items():
            # Remove the leading subtoken marker (id 6) if present
            if token_ids and token_ids[0] == 6:
                token_ids = token_ids[1:]

            token_ids_tuple = tuple(token_ids)
            ids2lemma[token_ids_tuple] = lemma
            lemma2ids[lemma].append(token_ids_tuple)

            # Track the length of the longest tokenized word
            max_token_length = max(max_token_length, len(token_ids))

        # Store the mappings and the longest token length as class attributes
        self.ids2lemma = ids2lemma
        self.lemma2ids = lemma2ids
        self.max_token_length = max_token_length


    def collect_usages(self, sentences):
        """
        Collect BERT representations for target words in sentences and count their occurrences.
        """
        warnings.filterwarnings("ignore")  # Suppress all warnings during this block
        
        target_counter = {lemma: 0 for lemma in self.lemma2ids}
        
        self.n_sentences = 0
        for sentence in sentences:
            self.n_sentences += 1
            
            # Tokenize the sentence using the tokenizer without adding special tokens
            sentence_token_ids = self.tokenizer.encode(' '.join(sentence), add_special_tokens=False)

            # Search for target words in the tokenized sentence
            while sentence_token_ids:
                candidate_ids_found = False
                # Iterate over token lengths from longest to shortest
                for length in range(self.max_token_length, 0, -1):
                    candidate_ids = tuple(sentence_token_ids[-length:])
                    if candidate_ids in self.ids2lemma:
                        # Increment the counter for the identified lemma
                        target_counter[self.ids2lemma[candidate_ids]] += 1
                        # Remove the matched sequence from the sentence tokens
                        sentence_token_ids = sentence_token_ids[:-length]
                        candidate_ids_found = True
                        break
                # If no match found, remove the last token and continue
                if not candidate_ids_found:
                    sentence_token_ids = sentence_token_ids[:-1]

        self.logger.info(f"Usages found: {sum(target_counter.values())}")
        for lemma, count in target_counter.items():
            self.logger.info(f'{lemma}: {count}')
        
        N_LAYERS = 1 # "top" or "average"
        self.embeddings = {t: np.empty((count, N_LAYERS * DIM)) for t, count in target_counter.items()}
        self.curr_idx = {t: 0 for t in target_counter}

    def collate(self, batch):
        """
        Collates a batch of input items into a format suitable for model training.
        """
        input_ids = torch.cat([item[0]['input_ids'] for item in batch], dim=0)
        attention_masks = torch.cat([item[0]['attention_mask'] for item in batch], dim=0)
        labels = [item[1] for item in batch]
        additional_info = [item[2] for item in batch]
        
        return [{'input_ids': input_ids, 'attention_mask': attention_masks}, labels, additional_info]
    
    def _update_embeddings(self, batch_lemmas, batch_target_pos, hidden_states):
        """
        Updates the embeddings with the hidden states from the model for each word's usage.
        """
        # Iterate over each item in the batch
        for b_id in np.arange(len(batch_lemmas)):
            lemma = batch_lemmas[b_id]

            # Extract hidden states for the given word's position in each layer
            layers = [layer[b_id, batch_target_pos[b_id] + 1, :] for layer in hidden_states]

            if LAYER == "top":
                emb_vector = layers[-1]
            elif LAYER == "average":
                emb_vector = np.mean(layers, axis=0)
            else:
                raise ValueError(f"Invalid method for extracting embeddings. Choose 'top', 'average'.")

            if emb_vector.shape[0] > 1:
                emb_vector = np.mean(emb_vector, axis=0)
            elif emb_vector.shape[0] == 0:
                emb_vector = np.full((768,), 0)
                
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
            batch_input_ids = batch_tuple[0] # Token IDs
            batch_lemmas = batch_tuple[1]  # Lemmas (target words)
            batch_target_pos = batch_tuple[2]    # Token positions for target words

            with torch.no_grad():
                if torch.cuda.is_available():
                    batch_input_ids['input_ids'] = batch_input_ids['input_ids'].to('cuda')
                    batch_input_ids['attention_mask'] = batch_input_ids['attention_mask'].to('cuda')

                # Forward pass through the model
                outputs = self.model(**batch_input_ids)

                hidden_states = [layer.detach().cpu().numpy() if torch.cuda.is_available() else layer.numpy() for layer in outputs.hidden_states]

                # Update embeddings with the hidden states for this batch
                self._update_embeddings(batch_lemmas, batch_target_pos, hidden_states)



    def run(self):
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.logger.info(f"Using device: {self.device}")

        self.set_seed(42)
        self._initialize_model()
        self._store_vocabulary()
    
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        sentences = LineSentence(self.data_path)
        self.collect_usages(sentences)

        dataset = MultipleTokenContext(self.ids2lemma, sentences, CONTEXT_SIZE, self.tokenizer, self.max_token_length, self.n_sentences)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=self.collate)

        # Process the batches
        self._process_batches(dataloader)

        # Save file
        self.embeddings = {
            w.split('_', 1)[0]: self.embeddings[w][~(self.embeddings[w] == 0).all(1)]
            for w in self.embeddings
        }
        np.savez_compressed(self.output_file, **self.embeddings)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Processing completed in {elapsed_time:.2f} seconds.")

























































