import tqdm
import json
import torch
import datasets
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
f_prompt = lambda x: tokenizer.apply_chat_template( \
        [{"role": "user", "content": base_prompt + '\n' + x + 'Summary:\n\n'}], \
        tokenize=False, add_generation_prompt=True)

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
    early_stopping=True
)
ds = datasets.load_from_disk('/home/ostovane/.cache/.dpo/dpo/sampled_dataset_test')
summaries = {}
batch_size = 8
for i in tqdm.trange(0, len(ds), batch_size):
    batch = ds[i:i+batch_size]
    prompts = [system + f_prompt(row) for row in batch['article']]
    tok = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    tok = {k: tok[k].to(device) for k in tok}
    with torch.no_grad():
        res = model.generate(**tok, generation_config=gen_config)
    decoded = tokenizer.batch_decode(res)
    for idx, summary in zip(batch['id'], decoded):
        summaries[idx] = summary
    
with open(model_name + "_summaries.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=4)

