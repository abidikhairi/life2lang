.PHONY: test

test:
	python -m pytest


pretrain:
	python scripts/train/pretrain.py --base_model google/flan-t5-small \
		--train_file "/run/media/khairi/Seagate Exp/datasets/life2lang/data/pretraining/dataset/train.csv"	\
		--valid_file "/run/media/khairi/Seagate Exp/datasets/life2lang/data/pretraining/dataset/validation.csv" \
		--output_dir "/tmp/life2lang-pretraining"