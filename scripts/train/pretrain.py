import os
import argparse

from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from millify import millify

from life2lang.utils import (
    print_gpu_info,
    print_gpu_memory,
    clean_gpu_memory,
    count_trainable_parameters,
    tokenize_examples
)

from life2lang.models import (
    T5ForConditionalGeneration,
    T5Tokenizer
)


def main(args):
    print_gpu_info()
    print_gpu_memory()
    
    base_model = args.base_model
    output_dir = args.output_dir
    
    train_data = Dataset.from_csv(args.train_file, sep="\t")
    valid_data = Dataset.from_csv(args.valid_file, sep="\t")
    
    print(f"Training data size: {len(train_data)}") # type: ignore
    print(f"Validation data size: {len(valid_data)}") # type: ignore
    print(f"Using base model: {base_model}")
    
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    
    print(f"Number of trainable parameters: {millify(count_trainable_parameters(model))}")

    # Ensure pad token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    valid_data = valid_data.map(lambda x: tokenize_examples(x, tokenizer), batched=True, batch_size=16) \
            .select_columns(['input_ids', 'attention_mask', 'labels'])
            
    train_data = train_data.map(lambda x: tokenize_examples(x, tokenizer), batched=True, batch_size=16) \
            .select_columns(['input_ids', 'attention_mask', 'labels'])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        
        num_train_epochs=5,
        eval_strategy='steps',
        eval_steps=100,
        save_steps=100,
        logging_steps=100,
        save_total_limit=2,
        report_to='none',
        run_name='pretraining-life2lang-small',
        
        remove_unused_columns=False,
        push_to_hub=False,
        hub_model_id='khairi/life2lang-small-pt',
        load_best_model_at_end=True,
        
        learning_rate=1.2e-3,
        optim='adamw_torch',
        weight_decay=0.01,
        max_grad_norm=0.1,
        warmup_ratio=0.15,
        lr_scheduler_type='cosine',
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data, # type: ignore
        eval_dataset=valid_data, # type: ignore
        data_collator=data_collator,
    )

    clean_gpu_memory()
    
    trainer.train()
    
    trainer.save_model(output_dir + "/final_model")
    tokenizer.save_pretrained(output_dir + "/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Life2Lang model.")
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model identifier from Hugging Face Model Hub."
    )
    
    ## Add train_file, valid_file
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the training data file."
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        required=True,
        help="Path to the validation data file."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model and checkpoints."
    )
    
    
    args = parser.parse_args()
    main(args)

