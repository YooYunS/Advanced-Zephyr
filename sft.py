import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from accelerate import Accelerator
from datasets import load_dataset, Dataset
import huggingface_hub
from typing import Literal
import re
import random

from trl import SFTTrainer

import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, help="Required to upload models to hub.")

    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset_name", type=str, default="ryan0712/ultra_no_robots")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--do_sample", type=bool, default=False, help="Sample the dataset.")
    parser.add_argument("--sample_size", type=int, default=None)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="epoch", help="You can choose the strategy of saving model.")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=True)
    parser.add_argument("--use_flash_attention", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="SFT/final_checkpoint"
    )
    parser.add_argument(
        "--hf_hub_path",
        type=str,
        help="The hub path to upload the model"
    )

    return parser.parse_args()

def apply_chat_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

        example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
        example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
    else:
        raise ValueError(
            f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example

def dataset_sampling(dataset, sample_size):
    random_indices = random.sample(range(len(dataset)), sample_size)

    random_samples = dataset.select(random_indices)

    return random_samples

def create_datasets(dataset_name, split):
    dataset = load_dataset(
        dataset_name,
        split=split,
    )
    if args.do_sample:
        dataset = dataset_sampling(dataset, args.sample_size)
    
    return dataset

if __name__ == "__main__":
    set_seed(42)
    args = arg_parse()

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # device_map={"": Accelerator().process_index},    # unavailable in deepspeed
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=args.use_flash_attention,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    special_tokens_dict = {"additional_special_tokens": ["<unk>", "<s>", "</s>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    train_dataset = create_datasets(args.dataset_name, args.train_split)
    # eval_dataset = create_datasets(args.dataset_name, args.test_split)
    
    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})
    eval_dataset = eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})
    
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size if eval_dataset else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim="adamw_torch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,  # True
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        packing=args.packing,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        neftune_noise_alpha=5,
        args=training_args,
    )

    train_result = trainer.train()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if accelerator.is_main_process:
        trainer.model.push_to_hub(args.hf_hub_path)
        trainer.tokenizer.push_to_hub(args.hf_hub_path)
    
    accelerator.wait_for_everyone()

    trainer.save_model(args.output_dir)
