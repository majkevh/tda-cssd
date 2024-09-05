import os
from gensim import utils as gensim_utils
import itertools
import torch
import warnings
from tqdm import tqdm


dict_languages = {
        "german": {
            "corpus1": "dta.txt",
            "corpus2": "bznd.txt",
            "ELMo-model": "201/",
            "BERT-model": "google-bert/bert-base-german-cased",
            "XLMR-model": "FacebookAI/xlm-roberta-base"
        },
        "english": {
            "corpus1": "ccoha1.txt",
            "corpus2": "ccoha2.txt",
            "ELMo-model": "209/",
            "BERT-model": "google-bert/bert-base-uncased",
            "XLMR-model": "FacebookAI/xlm-roberta-base"
        },
        "swedish": {
            "corpus1": "kubhist2a.txt",
            "corpus2": "kubhist2b.txt",
            "ELMo-model": "202/",
            "BERT-model": "af-ai-center/bert-large-swedish-uncased",
            "XLMR-model": "FacebookAI/xlm-roberta-base"
        },
        "latin": {
            "corpus1": "LatinISE1.txt",
            "corpus2": "LatinISE2.txt",
            "ELMo-model": "203/",
            "BERT-model": "google-bert/bert-base-multilingual-cased",
            "XLMR-model": "FacebookAI/xlm-roberta-base"
        },
}


class OneTokenContext(torch.utils.data.Dataset):
    """
    Generates context windows around a single token.
    It creates context sequences surrounding a specific token and aligns the token's position in the new context window.
    """

    def __init__(self, targets_i2w, sentences, context_size, tokenizer, n_sentences=None):
        """
        Initializes the OneTokenContext dataset for BERT-based models.

        :param targets_i2w: A dictionary mapping target token IDs to corresponding words.
        :param sentences: A list of tokenized sentences to extract contexts from.
        :param context_size: The desired size of the context window (e.g., 128, 256).
        :param tokenizer: A tokenizer used to encode the sentences into token IDs.
        :param n_sentences: Optional, the total number of sentences to process (useful for progress tracking).
        
        The constructor iterates through the provided sentences and identifies target tokens from `targets_i2w`.
        It then constructs a context window around each target token and records the necessary information.
        This class is particularly suited for single-token targets, such as those in BERT-based models.
        """
        super().__init__()
        self.data = []
        CLS_id, SEP_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0], tokenizer.encode('[SEP]', add_special_tokens=False)[0]

        def get_context(token_ids, target_pos, seq_len):
            window = (seq_len - 2) // 2
            start = max(0, target_pos - window)
            context_ids = token_ids[start:target_pos + window]
            padding = max(0, window - target_pos) + max(0, target_pos + window - len(token_ids))
            return context_ids + [0] * padding, target_pos - start

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sentence in tqdm(sentences, total=n_sentences):
                token_ids = tokenizer.encode(' '.join(sentence), add_special_tokens=False)
                for spos, tok_id in enumerate(token_ids):
                    if tok_id in targets_i2w:
                        context_ids, pos_in_context = get_context(token_ids, spos, context_size)
                        self.data.append(([CLS_id] + context_ids + [SEP_id], targets_i2w[tok_id], pos_in_context))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, lemma, pos_in_context = self.data[index]
        return torch.tensor(input_ids), lemma, pos_in_context



class MultipleTokenContext(torch.utils.data.Dataset):
    """
    Generates context windows around multi-token targets.
    It extracts the context around token spans (e.g., multi-token phrases) and adjusts the target's position within the context window.
    """

    def __init__(self, targets_i2w, sentences, context_size, tokenizer, max_token_len=10, n_sentences=None):
        """
        Initializes the MultipleTokenContext dataset for XLM-R based models.

        :param targets_i2w: A dictionary mapping target multi-token sequences to corresponding words or phrases.
        :param sentences: A list of tokenized sentences to extract contexts from.
        :param context_size: The desired size of the context window (e.g., 128, 256).
        :param tokenizer: A tokenizer used to encode the sentences into token IDs.
        :param max_token_len: Maximum number of tokens for any given target span.
        :param n_sentences: Optional, the total number of sentences to process (useful for progress tracking).
        
        The constructor iterates through the provided sentences, identifies target multi-token sequences from `targets_i2w`,
        and constructs context windows around each target sequence, recording relevant information such as the new token positions.
        This class is particularly suited for multi-token spans, such as those used in XLM-R models.
        """
        self.data = []
        self.tokenizer = tokenizer
        self.context_size = context_size

        def get_context(token_ids, target_pos, seq_len):
            window = (seq_len - 4) // 2
            target_len = target_pos[1] - target_pos[0]
            left_window = window - target_len // 2
            right_window = window - (target_len - 1) // 2

            start = max(0, target_pos[0] - left_window)
            end = min(len(token_ids), target_pos[1] + right_window + 1)
            padding = max(0, left_window - target_pos[0]) + max(0, (target_pos[1] + right_window + 1) - len(token_ids))

            context_ids = token_ids[start:end]
            context_ids = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id]
            new_target_pos = (target_pos[0] - start + 1, target_pos[1] - start + 1)

            return {
                'input_ids': context_ids + [tokenizer.pad_token_id] * padding,
                'attention_mask': [1] * len(context_ids) + [0] * padding
            }, new_target_pos

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sentence in tqdm(sentences, total=n_sentences):
                token_ids_full = tokenizer.encode(' '.join(sentence), add_special_tokens=False)
                token_ids = list(token_ids_full)

                while token_ids:
                    candidate_found = False
                    for length in range(min(max_token_len, len(token_ids)), 0, -1):
                        candidate_ids = tuple(token_ids[-length:])
                        if candidate_ids in targets_i2w:
                            target_pos = (len(token_ids) - length, len(token_ids))
                            context_ids, pos_in_context = get_context(token_ids_full, target_pos, context_size)
                            self.data.append((context_ids, targets_i2w[candidate_ids], pos_in_context))
                            token_ids = token_ids[:-length]
                            candidate_found = True
                            break
                    if not candidate_found:
                        token_ids = token_ids[:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        model_input, lemma, pos_in_context = self.data[index]
        model_input = {k: torch.tensor(v, dtype=torch.long).unsqueeze(0) for k, v in model_input.items()}
        return model_input, lemma, pos_in_context
