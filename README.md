# Evaluating Fairness and Generalizability of Large Language Models for Social Isolation Extraction from Unstructured Clinical Notes

This repository contains the code used in the paper “Evaluating Fairness and Generalizability of Large Language Models for Social Isolation Extraction”, which evaluates fairness and generalizability of Large Language Models.

## Overview

The script performs:
- Text preprocessing and cleaning of clinical narratives.
- Fine-tuning of FLAN-T5 for multi-class classification.
- Evaluation using macro F1 score and detailed classification reports.
- Evaluating Fairness and Generalizability using Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD).


### 1. Fine-tune FLAN-T5
```
python train_flant5.py
```

### 2. Evaluating Fairness and Generalizability
```
python flan_t5_generalizability_fairness_evaluation.py
```
