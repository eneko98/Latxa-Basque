from huggingface_hub import notebook_login
#notebook_login()

import json
import random
import os
import psutil

import torch
from torch.cuda import OutOfMemoryError

import accelerate
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    EarlyStoppingCallback,
    TextDataset, 
    DataCollatorForLanguageModeling, 
    IntervalStrategy,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
OUTPUT_DIR = "/home/pricie/cclstudent9/Master Thesis/Code/Training/results/latxa-basque"

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=1000,
            num_training_steps=num_training_steps
        )

"""Example sentences in the corpus are quite long so this
function selects the sentences in the corpus 
until the first full stop to avoid problems with the context window"""
def select_up_to_first_period(text):
    if not text:
        return ""
    end_index = text.find('. ')
    if end_index != -1:
        return text[:end_index + 1]
    else:
        return text
    
def load_dataset(train_path, val_path, tokenizer, max_seq_length=128):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=val_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, val_dataset, data_collator

with open('/home/pricie/cclstudent9/Master Thesis/Data/eeh_basque_corpus.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

processed_data = []
for entry in data:
    word = entry['word']
    pos = entry['pos']
    definition = entry['definition']
    examples = select_up_to_first_period(entry.get('examples', ''))

    processed_entry = f"[BOS] {word} (POS: {pos}) <definition> {definition} [EOS]"
    processed_data.append(processed_entry)

random.shuffle(processed_data)

split_index = int(0.9 * len(processed_data))
train_data = processed_data[:split_index]
val_data = processed_data[split_index:]

train_data_path = '/home/pricie/cclstudent9/Master Thesis/Code/Training/Data/EEH/train_data.txt'
val_data_path = '/home/pricie/cclstudent9/Master Thesis/Code/Training/Data/EEH/val_data.txt'


with open(train_data_path, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write("%s\n" % item)

with open(val_data_path, 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write("%s\n" % item)

model_id = "HiTZ/latxa-7b-v1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    trust_remote_code=True,)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

train_dataset, val_dataset, data_collator = load_dataset(train_data_path, val_data_path, tokenizer, max_seq_length=128)

lora_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    r=16,
    bias="none",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    auto_find_batch_size=True,
    #per_device_train_batch_size=8,
    #per_device_eval_batch_size=8,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=50,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    warmup_steps=1000,
    weight_decay=0.01,
    learning_rate=1e-3,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    output_dir=OUTPUT_DIR,
    report_to="all",
    run_name="eeh_latxa",
    load_best_model_at_end=True, 
    metric_for_best_model='loss',
    optim="paged_adamw_8bit"
    )


trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

print("Starting training with the custom cosine scheduler")
print("using GPUs 1 and 2")
try:
    trainer.train()
except OutOfMemoryError:
    print("Ran out of memory during training. Try reducing the batch size or sequence length.")
    torch.cuda.empty_cache()