import json
import random
import tqdm
from tqdm import tqdm

import numpy as np
import torch
    
def check_gpu():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f"There is {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))

    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
        
    return device

def load_data(filepath):
    """Given .jsonl file, returns a dictionary of the contents."""
    d = {}
    with open(filepath) as f:
        for i, line in tqdm(enumerate(f)):
            d[i] = json.loads(line)
    
    return d

#rouge scores for a reference/generated sentence pair
#source: google seq2seq source code.

import itertools

def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences."""

    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
    n: which n-grams to calculate
    text: An array of tokens
    Returns:
    A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def rouge_n(reference_sentences, evaluated_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    Args:
        evaluated_sentences: The sentences that have been picked by the summarizer
        reference_sentences: The sentences from the reference set
        n: Size of ngram.  Defaults to 2.
    Returns:
        recall rouge score(float)
    Raises:
        ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)

    reference_count = len(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    return overlapping_count/reference_count if reference_count else 0.0
