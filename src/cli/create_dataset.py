import re
import json
import datasets 
summary_files = [
    'Llama3.2-1b-base_summaries.json',
    'Llama3.2-1b-multiprompt_summaries.json',
    'Llama3.2-1b-without-multiprompt_summaries.json',
    'Llama3.2-3b-base_summaries.json',
    'Llama3.2-3b-multiprompt_summaries.json',
    'Llama3.2-3b-without-multiprompt_summaries.json',
    'pegasus_cnn_dailymail_summaries.json',
    ]
ds = datasets.load_from_disk('sampled_dataset_test/')

def f_sum(x):
    try:
        summary = js[x['id']]
        summary = re.sub('Here is a summary.*?\:', '', summary)
        x['summary'] = summary
    except:
        x['summary'] = ''

    return x

for file in summary_files:
    with open(file, 'r') as f:
        js = json.load(f)

    new_ds = ds.map(f_sum)
    new_ds.to_csv('csv_output/' + '.'.join(file.split('.')[:-1]) + '.csv')
