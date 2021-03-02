#### Tuesday, 2021.03.02

**current status**
- 700k "unique" result_comparison sentences involving ? manuscripts and ? cited papers
  - "unique" = cite only one paper
  - note: some sentences in fact cite multiple, but not all cite_spans reflect the correct number of cited papers, so I wasn't able to filter those sentences out
  - disregard this as noise for now

**notes**
- stick with wiki_unigrams.bin for now
- worry about the hyperparameter $k$ later

**next steps**
- [ ] finalise the dataset
- [ ] get summary statisics of finalised dataset
- [ ] perform 80-20 train-test split
- [ ] read REFRESH extractive summarisation paper
- [ ] understand how the ROUGE code works
- [ ] compute ROUGE-1, ROUGE-2, ROUGE-L, and F1 scores

---
#### Sunday, 2021.02.28

**current status**
- 1 million result_comparison sentences from ~500k manuscript papers and ~500k cited papers

**problems**
- issues with sent2vec
  - unable to download any of the .bin files provided at https://github.com/epfml/sent2vec
  - thus unable to vectorize sentences and compute distances
  - **update**: NL will be transferring the .bin files to zucchero for me
-

**next steps**
  - [x] evaluate ROUGE scores of top_cited_sentences against the gold citation_sentence as a preliminary evaluation of the quality of extractive summarization
  - [x] remove citation_sentences that reference more than one paper
    - problem: "cite_spans" fields in original s2orc do not necessarily contain all references, so it's difficult to correctly filter out sentences that contain more than one reference, e.g.
      - paper_id: 8281923
      - sentence: "Such a discrepancy may likely be due to dose-or species-specific differences since an acute administration of 3.3 mg/kg MDMA did not stimulate locomotion in mice (Scearce-Levie et al, 1999) , but did in rats ( Bankson and Cunningham, 2002)."
  - [x] switch to sent2vec
    - finally managed to download the bin files
    - transfer speed is stupid slow... I'll transfer the file personally on Monday
