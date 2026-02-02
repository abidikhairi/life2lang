import os
import argparse
import datetime
import logging
import torch
import pandas as pd
from Bio import SeqIO
from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TASK_ID_MAP = {
    1: ("predict protein localization sites: ", "cellular_localization"),
    2: ("predict biological processes from protein sequence: ", "biological_process"),
    3: ("predict molecular functions from protein sequence: ", "molecular_function"),
}


def main(args):
    NUMBER_OF_SEQUENCES = 224309  # cat test_file | grep ">" | wc -l
    test_file = args.test_file
    model_path = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading tokenizer from {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    task_prefix, function_aspect = TASK_ID_MAP[args.task_id]

    logger.info(f"Loading test file from {test_file}")
    logger.info(f"Task prefix: {task_prefix}")
    logger.info(f"Function aspect: {function_aspect}")

    rows = []
    idx = 0
    with open(test_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            if len(str(record.seq)) > 400:
                continue
            
            sequence = " ".join(tokenizer.tokenize(str(record.seq)))
            sequence = f"[seq] {sequence} [/seq]"
            text_input = task_prefix + "\n" + sequence

            input_ids = tokenizer(text_input, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(
                input_ids,
                max_length=128,
                num_beams=5,
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=3,
            )

            for i, (output_seq, output_score) in enumerate(
                zip(outputs.sequences, outputs.sequences_scores)
            ):
                output_score = output_score.exp().item()
                output_seq = tokenizer.decode(output_seq, skip_special_tokens=True)
                rows.append(
                    {
                        "id": record.id,
                        "target": output_seq,
                        "score": output_score,
                    }
                )
            idx += 1

            if idx % 1000 == 0:
                logger.info(f"Processed [{idx + 1}/{NUMBER_OF_SEQUENCES}] sequences")

    logger.info(f"Saving predictions to {args.output_dir}")
    today = datetime.datetime.now().strftime("%Y_%m_%d")

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(args.output_dir, f"predictions_{function_aspect}_{today}.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
