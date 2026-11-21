# MAMSA-MVSA-Single
# Multi-Attention Multimodal Sentiment Analysis (MAMSA) on MVSA-Single

PyTorch implementation and comparison of **MAMSA** (Wang et al., 2025) on the **MVSA-Single** multimodal sentiment dataset, evaluating:
- **CLIP MAMSA:** `openai/clip-vit-base-patch16` text and vision backbones projected into a shared 512-D space[cite: 1, 3].
- **BERT + CLIP MAMSA:** `bert-base-uncased` text encoder (with masked pooling and linear/LayerNorm projection to 512-D) paired with `openai/clip-vit-base-patch16` vision features[cite: 1, 2].

## Pipeline Overview
1. **Encoding:** Extracts text token sequences and visual patch tokens into a 512-D representation.
2. **Adaptive Attention Interaction (AAI):** Computes gating weights from pooled representations to guide cross-modal attention.
3. **Sentiment Association (SA):** Uses a learnable sentiment matrix initialized from positive, neutral, and negative prompts.
4. **Hierarchical Fusion (HF):** Combines sentiment features and cross-modal context sequentially[cite: 1].
5. **Classification Head:** Predicts 3-class sentiment (`Positive`, `Neutral`, `Negative`)[cite: 1, 2, 3].

## Data & Leakage Prevention
- Cleans MVSA-Single annotations (`labelResultAll*.txt`) by removing contradictory positive/negative pairs, resolving neutral pairs consistently, and dropping duplicate IDs[cite: 1].
- Splits data **80 / 10 / 10** (`seed=42`) and applies minority-class oversampling **only to the training split**[cite: 1].
- Selects best checkpoints using `0.5 * Val Acc + 0.5 * Val Macro F1`[cite: 1].
- *Note:* Raw dataset images and labels are not uploaded here; please obtain MVSA-Single from its official source and update the dataset path in the notebook/script.

## Experimental Results (452 Test Samples)

| Variant | Test Accuracy | Test Macro F1 |
| :--- | :---: | :---: |
| **BERT + CLIP MAMSA** | **73.01%**[cite: 2] | **59.59%**[cite: 2] |
| **CLIP MAMSA** | 71.46%[cite: 3] | 57.37%[cite: 3] |

### Per-Class Metrics
- **BERT + CLIP MAMSA:** Positive (`P: 0.77, R: 0.88, F1: 0.82, S: 266`), Neutral (`P: 0.38, R: 0.28, F1: 0.32, S: 43`), Negative (`P: 0.71, R: 0.59, F1: 0.64, S: 143`)[cite: 2].
- **CLIP MAMSA:** Positive (`P: 0.80, R: 0.79, F1: 0.80, S: 266`), Neutral (`P: 0.42, R: 0.19, F1: 0.26, S: 43`), Negative (`P: 0.62, R: 0.73, F1: 0.67, S: 143`)[cite: 3].

## Setup & Usage
```bash
pip install -r requirements.txt

