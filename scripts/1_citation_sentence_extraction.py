import json
import os
import re
import sys
import tqdm
from tqdm import tqdm 

import pandas as pd

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

###############################################################################

if __name__ == "__main__":
    paper_ids = set(line.strip() for line in open("./notebooks/paper_ids.txt"))
    
    punkt_params = PunktParameters()
    punkt_params.abbrev_types = set(["i.e", "e.g", "etc", "al", "fig", "figs", 
                                     "ref", "refs", "p", "c", "s"]) 
    tokenizer = PunktSentenceTokenizer(punkt_params)

    citation_sentences = []
    filepaths = [f"/home/jessica/data/s2orc/s2orc_shard{i:02d}.jsonl" for i in range(85)]
    for fpath in tqdm(filepaths):
        shard = []
        with open(fpath) as f:
            for line in f:
                shard.append(json.loads(line))

        for paper in shard:
            manuscript_id = paper["paper_id"]
            full_text = paper["body_text"]

            for paragraph in full_text:
                section_name = paragraph["section"].lower()
                if "discuss" not in section_name and "conclu" not in section_name:
                    continue 

                if not paragraph["cite_spans"]: 
                    continue

                paragraph_text = paragraph["text"]
                endpoints = list(tokenizer.span_tokenize(paragraph_text))

                j = 0
                for cite_span in paragraph["cite_spans"]:
                    cite_id = cite_span["cite_id"]
                    if cite_id not in paper_ids:
                        continue
                    
                    cite_text = cite_span["text"]
                    start, end = cite_span["start"], cite_span["end"]

                    a, b = endpoints[j]
                    while start >= b:
                        j += 1
                        a, b = endpoints[j]

                    textual = re.search('[a-zA-Z]', cite_text) 
                    if not (textual or re.search("[a-zA-Z]", paragraph_text[a:start])): 
                        a, b = endpoints[j-1]

                    citation_sentence = paragraph_text[a:b]
                    citation_sentences.append((citation_sentence, manuscript_id, cite_id))
                
    ## convert to pd.DataFrame
    citation_sentences = pd.DataFrame(
        citation_sentences, 
        columns=["citation_sentence", "manuscript_id", "cited_id"]).drop_duplicates()
    
    filepath = "./notebooks/citation_sentences.jsonl"
    citation_sentences.to_json(filepath, orient="records", lines=True)
    print(f"Extracted citation sentences are saved at {filepath}.")