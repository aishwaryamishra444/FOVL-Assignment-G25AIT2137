# FOVL Assignment: Multilingual Word Embeddings for Indian Languages

**Aishwarya Mishra** · USN / Roll No. **G25AIT2137** · PG Diploma in AI, IIT Jodhpur

300-dimensional **FastText skip-gram** embeddings with character 3–6-grams, trained **from scratch on IndicCorpV2-5k only**. They are evaluated on the word-similarity benchmark of Akhtar et al. (2017) for **Gujarati, Hindi, Marathi, Punjabi, Tamil, Telugu and Urdu**. OOV words get vectors composed from their character n-grams, so no word pair is ever dropped.

## Results (public benchmark, Spearman ρ)

| Gujarati | Hindi | Marathi | Punjabi | Tamil | Telugu | Urdu | **Macro** |
|---|---|---|---|---|---|---|---|
| 0.333 | 0.200 | 0.228 | 0.297 | 0.239 | 0.174 | 0.209 | **0.240** |

Coverage, OOV statistics, all 12 experiments, t-SNE plots and the error analysis are in the report.

## Repository contents (= submission ZIP)

| Path | What it is |
|---|---|
| `G25AIT2137_FOVL_Assignment.ipynb` | Full source code (executed, with outputs): data loading, preprocessing, training, OOV handling, evaluation, experiments, export, t-SNE, analysis, and a **self-contained inference API** (§13) for the private test set |
| `G25AIT2137_FOVL_Report.pdf` | Experimental report (LaTeX source in `report/`) |
| `embeddings/<Language>_embeddings.npy` | Required arrays of shape **(N, 2, 300)**, float32, original pair order (`[i,0]` = word 1, `[i,1]` = word 2) |
| `vocab/<Language>_vocab.tsv` | Vocabulary mapping: row `i` of the `.npy` ↔ `word1`, `word2` (+ in-vocabulary flags) |
| `vocab/word_to_vector.json` + `embeddings/all_words_300d.npy` | Every benchmark word → its row in a (V, 300) matrix |
| `model/compact/` | Trained model (int8-quantised word and n-gram matrices, 92 MB), loaded by `load_model()` |
| `model/final_config.json` | Final hyper-parameters |
| `outputs/` | Result tables (CSV) and the t-SNE figure |

## Generate embeddings for new / private-test words

Run the first cell, then **§13** of the notebook (needs only `model/compact/`, not the corpus):

```python
wv = load_model('model/compact')
v = get_embedding('पुस्तकालयों', wv)                              # any word, OOV included -> (300,)
arr = embed_pairs_file('Hindi-test.txt', wv, 'Hindi_test.npy')   # -> (N, 2, 300), order preserved
```

## Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute G25AIT2137_FOVL_Assignment.ipynb
```

The notebook downloads IndicCorpV2-5k from Hugging Face and the benchmark from GitHub. With the default flags it loads the saved model, which reproduces the reported numbers exactly. Set `TRAIN_FINAL=True` to retrain from scratch (≈17 min on 2 cores) or `RUN_EXPERIMENTS=True` to rerun all experiments (≈1 h). Multi-threaded training is not bit-for-bit deterministic, so a retrained model differs slightly.

**Data and models used:** IndicCorpV2-5k (training, the only corpus); Akhtar et al. (2017) word-similarity sets (evaluation only). No pre-trained embeddings or models of any kind.
