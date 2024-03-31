from datasets import load_dataset

def get_custom_dataset(dataset_config, tokenizer, split='train'):
    dataset = load_dataset('csv', data_files='./energy-dataset/energy_pandas.csv')

    prompt = (
        f"Convert the question into an SQL parse. Question:\n{{question}}\n---\nSQL parse:\n"
    )

    def apply_prompt_template(sample):
        return {
            "question": prompt.format(question=sample["natural_language"]),
            "parse": sample["parsed_utterance"],
        }

    dataset = dataset.map(apply_prompt_template)

    def tokenize_add_label(sample):
        question = tokenizer.encode(tokenizer.bos_token + sample["question"], add_special_tokens=False)
        parse = tokenizer.encode(sample["parse"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": question + parse,
            "attention_mask" : [1] * (len(question) + len(parse)),
            "labels": [-100] * len(question) + parse,
            }

        return sample

    dataset = dataset.map(tokenize_add_label)

    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

    return dataset[split]