
def tokenize_examples(examples, tokenizer):
    model_inputs = tokenizer(examples['input_text'], return_tensors='pt', padding=True)
    labels = tokenizer(examples['target_text'], return_tensors='pt', padding=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

