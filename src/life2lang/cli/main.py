from fire import Fire
import torch
from life2lang.utils.tasks import Task, TASK_PREFIXES
from life2lang.models import T5ForConditionalGeneration, T5Tokenizer


def run(task: Task, model_path: str, user_input: str, device: str = "cpu"):
    task = Task(task)
    device = torch.device(device)

    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    task_prefix = TASK_PREFIXES[task]
    input = f"{task_prefix}\n{user_input}"

    inputs = tokenizer(input, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.9,
        do_sample=True,
        top_k=950,
        top_p=0.95,
    )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for decoded_output in decoded_outputs:
        print(decoded_output.replace(" ", ""))


if __name__ == "__main__":
    Fire(run)
