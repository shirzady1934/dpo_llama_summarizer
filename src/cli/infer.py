<<<<<<< HEAD
=======
import re
>>>>>>> f30f7f8 (infernece model)
import tqdm
import json
import torch
import datasets
<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
device = 'cpu'
model_name = 'Llama3.2-3b-multiprompt'
checkpoint_path = '/home/ostovane/.cache/.dpo/dpo/dpo_correct/checkpoint-10000'
model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
system_message = {"role": "system", "content": "You are a helpful assistant that summarizes news articles."}
system = tokenizer.apply_chat_template([system_message], tokenize=False)
base_prompt = "Summarize the following news article into a short and informative paragraph:"
=======
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
device = 'cuda'
#model_name = 'Llama3.2-1b--without-multiprompt-fin'
model_name = 'Llama3.2-1b-base-fin'
checkpoint_path = 'unsloth/Llama-3.2-1B-Instruct'
#checkpoint_path = 'final-dpo-llama3.2-1b-without-multiprompt'
model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
#system_message = {"role": "system", "content": "You are a helpful assistant that summarizes news articles and just generate summaries."}
#system_message = {"role": "system", "content": "You are a helpful assistant that summarizes news articles into short, concise summaries without warnings or extra commentary"}
system_message = {
            "role": "system",
                "content": "You are a helpful assistant that produces concise and accurate summaries of news articles. Focus on preserving key facts and main points while keeping the output short and clear."
                }

system = tokenizer.apply_chat_template([system_message], tokenize=False)
#base_prompt = "Summarize the following article:"
#base_prompt = "Summarize the following article in one short, factual sentence:"
base_prompt = "Write a brief summary (like a news highlight) for the following article. Capture the most important facts:"
f_prompt = lambda x: tokenizer.apply_chat_template( \
        [{"role": "user", "content": base_prompt + '\n' + x + 'Summary:\n\n'}], \
        tokenize=False, add_generation_prompt=True)

refusal_phrases = [
            "I can't", "I can_t", "I can~@~Yt", "cannot", "unable", "I'm sorry", "I am sorry",
                "I will not", "I do not", "I'm not able", "not allowed", "not able", "refuse",
                    "against policy", "glorifies", "promotes", "encourages", "animal cruelty",
                        "illegal", "harmful", "terrorism", "violence", "prohibited", "Can I help you"
                        ]


bad_words_ids = [tokenizer.encode(phrase, add_special_tokens=False) for phrase in refusal_phrases]

gen_config = GenerationConfig(
    do_sample=False,               # Deterministic output
    num_beams=4,                   # Use beam search for better quality
    temperature=1.0,               # Not used if do_sample=False
    top_p=1.0,                     # Not used if do_sample=False
    top_k=50,                      # Not used if do_sample=False
    repetition_penalty=1.05,       # Mild penalty to reduce repetition
    length_penalty=1.0,            # Neutral
    max_new_tokens=300,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    early_stopping=True,
    bad_words_ids=bad_words_ids
)
)
ds = datasets.load_from_disk('/home/ostovane/.cache/.dpo/dpo/sampled_dataset_test')
summaries = {}
batch_size = 8


gen_dict = gen_config.to_dict()
gen_dict['bad_words_ids'] = bad_words_ids
ds = datasets.load_from_disk('/home/ostovane/.cache/.dpo/dpo/sampled_dataset_test')#.select(range(50,55))
summaries = {}
batch_size = 10
for i in tqdm.trange(0, len(ds), batch_size):
    batch = ds[i:i+batch_size]
    prompts = [system + f_prompt(row) for row in batch['article']]
    tok = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    tok = {k: tok[k].to(device) for k in tok}
    with torch.no_grad():
        res = model.generate(**tok, generation_config=gen_config)
    decoded = tokenizer.batch_decode(res)
    for idx, summary in zip(batch['id'], decoded):
        #res = model.generate(**tok, generation_config=gen_config)
        res = model.generate(**tok, **gen_dict)

    generated_texts = []
    for i in range(len(prompts)):
        input_ids = tok['input_ids'][i]
        generated_ids = res[i]
        generated_only_ids = generated_ids[len(input_ids):]
        generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
        generated_text  = re.sub("Here's a summary of the article:", "", generated_text)
        generated_text  = re.sub("Here is a summary of the article:", "", generated_text)
        generated_texts.append(generated_text)

    for idx, summary in zip(batch['id'], generated_texts):
        if "create content that" in summary:
            continue
        summaries[idx] = summary
    
with open(model_name + "_summaries.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=4)

