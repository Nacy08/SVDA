import argparse
import inspect
import math
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


PROMPT = """You are the person described in the scenario.
Answer the question from your own perspective.
Your response should be natural, meaningful, and aligned with the target human value.
Do not merely repeat the value name. Do not explain the value label. Output only the response.

Scenario: {Scenario}
Question: {Question}
Target value: {Value}

Response:"""


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_example(example):
    prompt = PROMPT.format(
        Scenario=example.get("Scenario", ""),
        Question=example.get("Question", ""),
        Value=example.get("Value", ""),
    )
    answer = example.get("Consistent Value Response", "")
    return {"text": f"{prompt} {answer}"}


def steps_per_epoch_ratio(num_samples, batch_size, grad_accum, ratio):
    updates_per_epoch = math.ceil(num_samples / max(1, batch_size * grad_accum))
    return max(1, math.ceil(updates_per_epoch * ratio))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="track2_qlora_sft/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    train_file = Path(cfg["train_file"])
    eval_file = Path(cfg.get("eval_file", ""))
    data_files = {"train": str(train_file)}
    if eval_file and eval_file.exists():
        data_files["validation"] = str(eval_file)

    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True)
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules"),
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    save_steps = steps_per_epoch_ratio(
        len(dataset["train"]),
        cfg.get("per_device_train_batch_size", 1),
        cfg.get("gradient_accumulation_steps", 8),
        cfg.get("eval_ratio_per_epoch", 0.5),
    )
    has_eval = "validation" in dataset

    training_kwargs = dict(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=cfg.get("save_total_limit", 8),
        eval_steps=save_steps if has_eval else None,
        report_to="none",
        remove_unused_columns=False,
    )
    training_kwargs.update(
        dataset_text_field="text",
        max_length=cfg.get("max_seq_length", 512),
        packing=False,
    )
    eval_strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(SFTConfig.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[eval_strategy_name] = "steps" if has_eval else "no"
    training_args = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(cfg["final_adapter_dir"])
    tokenizer.save_pretrained(cfg["final_adapter_dir"])


if __name__ == "__main__":
    main()
