import os
import csv
import re
import html
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint_path = "/home/pricie/cclstudent9/Master Thesis/Code/Training/results/latxa-basque/checkpoint-500"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

rouge_metric = load_metric("rouge")
bertscore_metric = load_metric("bertscore")

def generate_definition(model, tokenizer, word, pos, max_length=150, top_p=0.9, temperature=0.9, num_beams=3, do_sample=True):
    prompt = f"[BOS] {word} (POS: {pos}) <definition>"

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=do_sample,
        top_k=50,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    text = tokenizer.decode(output[0], skip_special_tokens=False)

    text = text.replace("<s>", "").replace("[BOS]", "").replace("[EOS]", "").strip()

    start_index = text.find('<definition>') + len('<definition>')
    end_index = text.find('. <examples>') if '. <examples>' in text else len(text)
    
    definition = text[start_index:end_index].strip()
    definition = html.unescape(definition)
    definition = re.sub(r'^\d+\s|\d+\.\s', '', definition)  # Removes numbers that appear at the start of definitions
    definition = re.sub(r"\.\.+", ".", definition)  # Replace multiple dots with one dot
    definition = "Definition: " + definition

    return definition

csv_file_path = "/home/pricie/cclstudent9/Master Thesis/Code/Evaluation/EEH/evaluation_results.csv"

with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    if file.tell() == 0:
        csv_writer.writerow(["Word", "POS", "Generated Definition", "Expected Definition", "ROUGE-L F1", "BERTScore F1"])

    while True:
        word = input("Enter the word (or 'quit' to stop): ")
        if word.lower() == 'quit':
            break
        pos = input("Enter the part of speech: ")
        
        generated_definition = generate_definition(model, tokenizer, word, pos)
        print(f"Generated Definition: {generated_definition}\n")

        expected_definition = input("Enter the expected definition for metric calculation: ")

        rouge_score = rouge_metric.compute(predictions=[generated_definition], references=[expected_definition])
        bertscore_result = bertscore_metric.compute(predictions=[generated_definition], references=[expected_definition], lang="eu")

        print(f"ROUGE-L F1: {rouge_score['rougeL'].mid.fmeasure}")
        print(f"BERTScore F1: {bertscore_result['f1'][0]}")
        print("\n---\n")

        csv_writer.writerow([
            word,
            pos,
            generated_definition,
            expected_definition,
            rouge_score['rougeL'].mid.fmeasure,
            bertscore_result['f1'][0]
        ])
        file.flush()

print(f"Results saved to {csv_file_path}")
