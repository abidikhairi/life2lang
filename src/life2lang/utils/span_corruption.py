import random
from typing import List, Union


def protein_span_corruption(
        sequences: Union[str, List[str]],
        mean_span_length: int = 3,
        noise_density: float = 0.15
    ):
    """
    sequence: the protein sequence
    mean_span_length: average number of tokens per masked span
    noise_density: fraction of tokens to mask
    """

    if isinstance(sequences, str):
        tokens = list(sequences)
        return _corrupt_sequence(tokens=tokens, mean_span_length=mean_span_length, noise_density=noise_density)

    ret_seqs = []
    for seq in sequences:
        ret_seqs.append(
            _corrupt_sequence(tokens=list(seq), mean_span_length=mean_span_length, noise_density=noise_density)
        )
    
    ret_dict = {
        "input_text": [r['input_text'] for r in ret_seqs],
        "target_text": [r['target_text'] for r in ret_seqs]
    }
    return ret_dict



def _corrupt_sequence(
        tokens: List[str],
        mean_span_length: int = 3,
        noise_density: float = 0.15
    ):
    n_tokens = len(tokens)
    n_mask = int(round(n_tokens * noise_density))
    spans = []

    while sum(spans) < n_mask:
        span_len = max(1, int(random.expovariate(1.0 / mean_span_length)))
        spans.append(span_len)

    spans[-1] -= max(0, sum(spans) - n_mask)

    available_positions = list(range(n_tokens))
    random.shuffle(available_positions)
    start_positions = sorted(available_positions[:len(spans)])

    corrupted, targets = [], []
    last_idx = 0
    for i, start in enumerate(start_positions):
        end = min(start + spans[i], n_tokens)
        corrupted.extend(tokens[last_idx:start])
        corrupted.append(f"<extra_id_{i}>")
        targets.append(f"<extra_id_{i}> " + " ".join(tokens[start:end]))
        last_idx = end

    corrupted.extend(tokens[last_idx:])
    target_str = " ".join(targets).replace('  ', ' ')
    input_str = " ".join(corrupted).replace('  ', ' ')

    return {
        "input_text": input_str,
        "target_text": target_str
    }
