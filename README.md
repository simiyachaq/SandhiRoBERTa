# SandhiRoBERTa

A small encoder‑only transformer trained on sub‑phonemic feature streams of Sanskrit. The model is designed to absorb external *sandhi* (euphonic combination) as a local feature‑propagation task, so that a downstream syntactic fine‑tuning stage can operate on phonologically regularised word representations. The final product is intended as a tool for semi‑automatic linguistic annotation and for exploring construction‑grammar patterns that link phonology to syntax.

## Motivation

Sanskrit orthography applies *sandhi* across word boundaries. The surface form of a word therefore depends on its neighbour, producing a large number of allomorphic variants. Standard sub‑word tokenizers (BPE, WordPiece) fragment words into pieces that do not correspond to morphological units, while rule‑based sandhi splitters require extensive lexical resources and fail on out‑of‑vocabulary items.

SandhiRoBERTa approaches the problem from the opposite direction: instead of undoing sandhi, it represents every phoneme as a small set of articulatory features. Sandhi then becomes a local operation on one or two feature dimensions (e.g., voicing assimilation). A Transformer can learn these operations during masked language modelling, yielding representations that are already “sandhi‑free” before any syntactic information is provided.

The project draws inspiration from **Shan RoBERTa**, a parallel effort that demonstrated the failure of BPE tokenizers for isolating languages and the effectiveness of word‑level pre‑training for syntactic annotation. SandhiRoBERTa applies a similar philosophy but at the sub‑phonemic level, because Sanskrit’s fusional morphology and pervasive sandhi make the *feature* the appropriate atomic unit rather than the word.

## Approach

1. **Deterministic feature extraction** – A Python script converts Devanāgarī text into a linear sequence of phonemes. Each phoneme is decomposed into five discrete channels:
   - `punto` – place of articulation (13 real values; 0 = padding).
   - `manera` – manner of articulation (12 real values).
   - `sonoridad` – voicing / nasality (5 real values, with a dedicated value for nasalised vowels triggered by *candrabindu*).
   - `frontera_palabra` – word‑boundary tag (1 = interior, 2 = beginning, 3 = end; 0 = padding).
   - `inicio_silaba` – syllable‑onset flag (1 = not onset, 2 = onset; 0 = padding).

   All real values start at 1 so that 0 is reserved strictly for padding. The mask token for each channel is allocated one position beyond the highest real value, ensuring no overlap.

2. **Multi‑stream embedding** – Each channel has its own `nn.Embedding`. The five embeddings are concatenated and projected to the Transformer dimension (`d_model = 384`).

3. **Multi‑channel MLM pre‑training** – Channels are masked *independently* (e.g., only voicing may be hidden while place and manner remain visible). The model must predict the missing feature from the surrounding phonetic context. This teaches it the regularities of sandhi without any explicit rules. The `frontera_palabra` and `inicio_silaba` channels are also masked and predicted.

4. **Frozen encoder + syntactic heads** – After pre‑training, the encoder weights are frozen. Multiple linear classifiers are added on top of the hidden state of each phoneme token. They are trained on an annotated treebank to predict detailed morphological attributes (case, gender, number, person, tense, mood, verb form, voice) and the dependency relation (including *kāraka* roles). The loss is a simple sum of cross‑entropies over all heads, computed at every non‑padding position.

## Tokeniser details

The tokeniser (`scripts/tokenizador_infrasegmental.py`) converts raw Devanāgarī into the compressed TSV `corpora/sandhi_features.tsv.gz`. Its behaviour:

- **Inherent vowel *a***: emitted after a consonant unless a *virāma* (्) or a *mātrā* (dependent vowel sign) follows.
- **Consonant conjuncts** (e.g., *kt*, *ṣṭ*): only the first consonant of the cluster receives the syllable‑onset flag (`inicio_silaba = 2`); the remaining consonants in the cluster get `1`.
- **Diphthongs *ai* and *au***: expanded into two phonemes – a low central vowel followed by a palatal/labial off‑glide. The off‑glide never carries the syllable‑onset flag, distinguishing it from a true consonantal *y* or *v*.
- **Anusvāra (ं)**: a separate token with its own feature triple (place = anusvāra, manner = nasal, voicing = nasal).
- **Candrabindu (ँ)**: does **not** produce a token. Instead, it modifies the immediately preceding phoneme by setting its `sonoridad` to the value “nasalised” (index 5), reflecting the Pāṇinian definition of *anunāsika*.
- **Independent vowels** (e.g., अ, ए): they do **not** suppress the inherent vowel of a preceding consonant; both the inherent *a* and the independent vowel are emitted as distinct tokens, with the independent vowel always starting a syllable.

The feature tables distinguish:

- Close vowels (*i*, *ī*, *u*, *ū*) from mid vowels (*e*, *o*) by assigning them different place‑of‑articulation indices.
- Rhotic liquids (*r*) from lateral liquids (*l*, *ḷa*) by different manner indices.

The final tables use contiguous integer blocks so that no real category collides with the mask token. The mapping from characters to feature triples is fully deterministic and shared across the tokeniser, the dictionary builder, and the fine‑tuning data preparation scripts.

## Corpus sizes

- **Pre‑training corpus** (`train.txt`): derived from the Sanskrit Text Corpus for LLM Pre‑training, it contains approximately 237 million characters (≈646 MB). After tokenisation, it yields **2,585,103 sentences** and about **193 million phoneme tokens**, stored in `sandhi_features.tsv.gz` (≈250 MB compressed).
- **Fine‑tuning corpus** (`finetune_features.tsv`): built from the UD Sanskrit‑Vedic treebank and additional unannotated sentences, it contains **22,058 sentences** and **886,800 phoneme‑rows**. Every phoneme of a word carries the word’s full morphological and dependency labels.
- **Word‑feature dictionary** (`word_feature_dict.pkl.gz`): constructed from approximately **2.9 million unique surface word forms** found in the pre‑training corpus.

## Model architecture and pre‑training

- **Encoder**: RoBERTa‑style, 6 layers, 8 attention heads, hidden size 384, intermediate size 1536. ≈12 M parameters.
- **Embedding**: `MultiStreamEmbedding` with 5 independent embedding tables (dimension 32 each), concatenated and projected to 384.
- **Masking**: per‑channel, 15 % probability, 80‑10‑10 rule applied independently to each channel. The mask token is the last index in the vocabulary.
- **Training**: 30 epochs, batch size 64 with gradient accumulation (effective 128), AdamW (lr=3e‑4, weight decay 0.1), cosine schedule with 2 000 warm‑up steps, mixed precision (AMP). Training is performed on a single NVIDIA T4 GPU (Colab).
- **Checkpointing**: every 2 000 optimizer steps, saving model state, optimizer, scheduler, and the exact batch position within the epoch, allowing precise resumption.

The pre‑training loss on the feature streams decreases rapidly from ~2.4 to ~0.3 in the first few epochs, reflecting the highly systematic nature of sandhi and phonotactic rules.

## Fine‑tuning data preparation

The Universal Dependencies treebank `UD_Sanskrit‑Vedic` (file `sa_vedic-ud-train.conllu`) is used for fine‑tuning. A dedicated script (`scripts/conllu_to_features.py`) converts the CoNLL‑U into a TSV that aligns with the pre‑training representation:


- Each word’s surface form (in IAST) is transliterated to Devanāgarī and tokenised with exactly the same `process_word` function.
- Word boundaries are marked as in the pre‑training TSV.
- Every phoneme token of a word receives the word’s full set of linguistic labels (morphological attributes and dependency relation). Padding positions (index 0) are set to 0 and are later mapped to `ignore_index = -100` during training; all other positions contribute to the loss.
- The morphological annotation (stored in the `FEATS` column when `XPOS` is empty) is decomposed into eight separate attributes: `Case`, `Gender`, `Number`, `Person`, `Tense`, `Mood`, `VerbForm`, and `Voice`. Each attribute has its own vocabulary and becomes a separate column in the TSV.
- The dependency relation (`DEPREL`) is added as a ninth label column.
- Sentences are processed as atomic syntactic units based on the CoNLL‑U `# sent_id` boundaries. The maximum sequence length is expanded to 512 phoneme tokens to ensure 100% of Vedic sentences are preserved intact, allowing the self-attention mechanism to fully capture long-distance syntactic dependencies and constructional tmesis without sequence truncation.

Separate vocabulary files (`Case_vocab.json`, `Gender_vocab.json`, …, `deprel_vocab.json`) are saved alongside the TSV.

## Fine‑tuning model

The fine‑tuning model (`SandhiSyntacticTagger`) consists of:
- The frozen pre‑trained encoder (embedding + Transformer layers).
- Nine independent linear heads, each projecting the hidden state of any phoneme token to the appropriate number of classes.

During training, only the heads are updated. The loss is the sum of cross‑entropies over all heads, computed at every non‑padding position. Because every phoneme carries the word’s grammatical labels, the model learns to associate sub‑phonemic feature streams with morphological and syntactic functions throughout the entire sequence. The encoder remains frozen, and the heads add only a few hundred thousand parameters.



## Construction grammar perspective

The architecture supports descriptive work in the framework of Construction Grammar (CxG). Because the model operates on sub‑phonemic features and learns to predict morphological and syntactic labels directly from surface forms, it can serve as a tool for inducing a constructicon from annotated data. The frozen encoder provides phonologically regularised word representations that abstract away from sandhi alternations; clustering these representations groups words by their constructional properties rather than by lexicographic convention. The approach is designed to capture form‑meaning pairings — including particle‑verb combinations, possessive expressions, and case‑marked argument structures — without resorting to traditional Pāṇinian derivational categories.

## Project structure

```
SandhiRoBERTa/
├── corpora/
│   ├── train.txt                     # raw corpus (not stored)
│   ├── sandhi_features.tsv.gz        # pre‑processed features (≈250 MB)
│   ├── word_feature_dict.pkl.gz      # word ↔ features dictionary
│   ├── sa_vedic-ud-train.conllu     # Vedic treebank
│   └── finetune_features.tsv         # fine‑tuning data with linguistic labels
├── scripts/
│   ├── tokenizador_infrasegmental.py  # feature tokeniser (parallel)
│   ├── crear_diccionario.py          # dictionary builder
│   ├── conllu_to_features.py         # CoNLL‑U → fine‑tuning TSV
│   ├── analizar_colisiones.py        # collision analysis (debug)
│   └── decode_features.py            # feature → IAST decoder
├── notebooks/
│   ├── sandhiroberta_pretrain.ipynb  # pre‑training notebook
│   └── sandhiroberta_finetune.ipynb  # fine‑tuning notebook
└── README.md
```

## Known limitations

- The model operates on surface forms with sandhi already applied; it does not recover underlying forms.
- The *avagraha* (ऽ) and the short vowel *a* (अ) are phonetically identical in our feature set and are not distinguished.
- Vowel nasalisation from *candrabindu* is represented as a single feature value on the modified vowel; the exact realisation (homorganic nasal consonant vs. pure vowel nasalisation) is not modelled explicitly.
- The corpus used for pre‑training may contain typographical inconsistencies that propagate into the feature dictionary.
- Fine‑tuning has only been structurally defined; empirical results on syntactic tasks are pending.

## Usage

1. Prepare the feature file:
   ```bash
   python3 scripts/tokenizador_infrasegmental.py
   ```
2. Build the word‑feature dictionary:
   ```bash
   python3 scripts/crear_diccionario.py
   ```
3. (Optional) Prepare fine‑tuning data:
   ```bash
   python3 scripts/conllu_to_features.py corpora/sa_vedic-ud-train.conllu
   ```
4. Upload `sandhi_features.tsv.gz` (and optionally `finetune_features.tsv`) to Google Drive, mount it in Colab, and run the provided notebooks.

## Acknowledgments

- The raw Sanskrit corpus is derived from the Kaggle dataset by Preet Sojitra, originally compiled by Kartik Bhatnagar.
- The Vedic treebank is from Universal Dependencies (UD_Sanskrit‑Vedic).
- The project structure and the two‑phase strategy are directly informed by the Shan RoBERTa project, which tackled analogous problems for a low‑resource isolating language.
