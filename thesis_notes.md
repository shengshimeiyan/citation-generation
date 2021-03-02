**date**: Sunday, 28.02.2021

**current status**:
- 1 million result_comparison sentences from ~500k manuscript papers and ~500k cited papers
- 700k "unique" result_comparison sentences 

**problems**:
- issues with sent2vec
  - unable to download any of the .bin files provided at https://github.com/epfml/sent2vec
  - thus unable to vectorize sentences and compute distances
  - **update**: NL will be transferring the .bin files to zucchero for me
-

**next steps**:
  - [x] evaluate ROUGE scores of top_cited_sentences against the gold citation_sentence as a preliminary evaluation of the quality of extractive summarization
    - implementation of ROUGE taken from some google code lol
  - [x] remove citation_sentences that reference more than one paper
    - currently keeping only result_comparison sentences that appear precisely once
    - problem: "cite_spans" fields in original s2orc do not necessarily contain all references, so it's difficult to correctly filter out sentences that contain more than one reference, e.g.
      - paper_id: 8281923
      - sentence: "Such a discrepancy may likely be due to dose-or species-specific differences since an acute administration of 3.3 mg/kg MDMA did not stimulate locomotion in mice (Scearce-Levie et al, 1999) , but did in rats ( Bankson and Cunningham, 2002) ."
  - [ ] switch to sent2vec
    - finally managed to download wiki_unigrams (wiki_bigrams still fails)
    - stupid low transfer speed... I'll transfer the file personally on Monday
