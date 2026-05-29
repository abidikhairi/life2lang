# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`life2lang` is a PhD research project that frames protein biology tasks as text-to-text problems using a T5 encoder-decoder model. Protein sequences are treated as "language" — the model is pretrained via span corruption and fine-tuned for tasks like function prediction, family classification, localization prediction, and protein design.

## Commands

**Install (editable mode):**
```bash
pip install -e ".[testing]"
```

**Run all tests:**
```bash
make test
# or directly:
python -m pytest
```

**Run a single test:**
```bash
python -m pytest tests/test_skeleton.py::test_name -v
```

**Pretrain:**
```bash
make pretrain
# or:
python scripts/train/pretrain.py \
  --base_model google/flan-t5-small \
  --train_file <path/to/train.csv> \
  --valid_file <path/to/validation.csv> \
  --output_dir <output_dir>
```

**Fine-tune:**
```bash
python scripts/train/finetune.py \
  --base_model <model_path_or_hf_id> \
  --train_file <path/to/train.tsv> \
  --valid_file <path/to/valid.tsv> \
  --output_dir <output_dir>
```

**Run inference via CLI:**
```bash
python -m life2lang.cli.main \
  --task <task_name> \
  --model_path <model_path> \
  --user_input "<protein sequence>" \
  --device cpu
```

**CAFA evaluation:**
```bash
python scripts/eval/cafa6.py \
  --test_file <fasta_file> \
  --model_path <model_path> \
  --task_id <1|2|3> \
  --output_dir <output_dir>
```

## Architecture

### Task framing

All biology tasks are text-to-text: inputs are prefixed with a natural language instruction (defined in `src/life2lang/utils/tasks.py`), and the model generates the output as free text. The five supported tasks are `FAMILY_GENERATION`, `FAMILY_CLASSIFICATION`, `BIOPROCESS_PREDICTION`, `LOCALIZATION_PREDICTION`, and `FUNCTION_PREDICTION`.

Protein sequences are wrapped with `[seq] ... [/seq]` markers at inference time (datamodule and CAFA eval script) but the CLI feeds them raw with a task prefix.

### Pretraining

Pretraining uses **span corruption** on raw protein sequences (`src/life2lang/utils/span_corruption.py`): random spans are replaced with `<extra_id_N>` sentinel tokens (T5-style), and the model learns to reconstruct them. The `ProteinSequenceDataModule` (`src/life2lang/datamodules/unsupervised_datamodule.py`) wraps this for PyTorch Lightning. Training script (`scripts/train/pretrain.py`) uses HuggingFace `Seq2SeqTrainer` directly instead of Lightning.

### Fine-tuning

Fine-tuning data is TSV with `input_text` / `target_text` columns. `tokenize_examples` in `src/life2lang/utils/dataset_utils.py` tokenizes both columns, setting `labels` to the tokenized target. The same `Seq2SeqTrainer` setup is reused with cosine LR schedule and AdamW.

### Model

The T5 implementation in `src/life2lang/models/t5/` is a local copy of the HuggingFace T5 implementation (not imported from transformers, allowing custom modifications). It uses `T5Config`, a custom `T5Tokenizer`, and exposes `T5ForConditionalGeneration` and `T5Model`. The public model interface mirrors HuggingFace's — `from_pretrained` and `save_pretrained` work normally.

### CAFA evaluation

`scripts/eval/cafa6.py` reads FASTA files, skips sequences longer than 400 residues, and runs beam search (5 beams, 3 returned sequences) for task IDs 1–3 (localization, biological process, molecular function). Outputs a CSV with `id`, `target`, `score` columns.
