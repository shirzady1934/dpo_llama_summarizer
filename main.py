import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
import pandas as pd

from trl import DPOConfig, DPOTrainer


tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv('generated_summaries_with_variants32.csv')

prompts = []
chosens = []
rejecteds = []
template_prompts = [
    'Read the following article and summarize it as if you are writing the lead paragraph for a news report:',
    'Summarize the following news article into a short and informative paragraph:',
    'Write a brief summary (like a news highlight) for the following article. Capture the most important facts:',
    'From the following news article, extract the key points a reader should know:',
]
models_list = ['Model 1 (llama-3.1-70b-versatile) Summary', 'Model 2 (gemma-9b-it) Summary', 'Model 3 (mixtral) Summary']
system_message = {"role": "system", "content": "You are a helpful assistant that summarizes news articles."}
system = tokenizer.apply_chat_template([system_message], tokenize=False)
prompt_message = lambda x: {"role": "user", "content": x}
for i in range(len(df)):
    for idx in models_list:
        for temp in template_prompts:
            #prompts.append(system_prm + temp + '\n' + df.loc[i,'Article'] + 'Summary:<|eot_id|>')
            message = prompt_message(temp + '\n' + df.loc[i,'Article'] + 'Summary:\n\n')
            prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            prompts.append(system + prompt)
            chosens.append(df.loc[i,'Reference Summary'])
            rejecteds.append(df.loc[i,idx])


data = {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
}
#data = {
#        "prompt": prompts[:100],
#        "chosen": chosens[:100],
#        "rejected": rejecteds[:100],
#}


dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(0.05)
print(dataset)




# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=8192, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=8192, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )


def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    cache_dir: Optional[str] = None,
    num_proc=32,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': list[str],
        'chosen': list[str],
        'rejected': list[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    
    def return_prompt_and_responses(samples) -> dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    if data_dir == 'data/rl':
        return dataset['train']
    else:
        return dataset['test']


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        'unsloth/Llama-3.2-3B-Instruct',
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        #load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]


    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_stack_exchange_paired(data_dir="data/rl")
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
        num_proc=32,
    )
    print(train_dataset)

    # 3. Load evaluation dataset
    eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation")
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
        num_proc=32,
    )
    print(train_dataset[0])
    print(eval_dataset)
    # 4. initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        beta=script_args.beta,
        loss_type='sigmoid',
        max_length=8192,
        max_prompt_length=4096,
        remove_unused_columns=False,
        run_name="dpo_llama2",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        #beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        #max_prompt_length=script_args.max_prompt_length,
        #max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

