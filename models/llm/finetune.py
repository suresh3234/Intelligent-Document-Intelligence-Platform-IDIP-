import os
import argparse
import logging
from typing import Dict, Any

import torch
try:
    import mlflow
except ImportError:
    mlflow = None

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Monkeypatch top_k_top_p_filtering in transformers module to ensure compatibility
# with TRL 0.7.11 which expects this attribute in newer transformers v4.45+ versions
import transformers
if not hasattr(transformers, "top_k_top_p_filtering"):
    try:
        from transformers.generation.utils import top_k_top_p_filtering
        transformers.top_k_top_p_filtering = top_k_top_p_filtering
    except ImportError:
        try:
            from transformers.generation.logits_process import top_k_top_p_filtering
            transformers.top_k_top_p_filtering = top_k_top_p_filtering
        except ImportError:
            def dummy_top_k_top_p_filtering(*args, **kwargs):
                pass
            transformers.top_k_top_p_filtering = dummy_top_k_top_p_filtering

from trl import SFTTrainer

logger = logging.getLogger("idip.models.llm.finetune")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning pipeline for IDIP's LLM component.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/training_data.jsonl",
        help="Local path to JSONL dataset or HF Dataset name."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HF repository name of the base LLM model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/llm/checkpoints",
        help="Directory to save the checkpoints and adapters."
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default="models/llm/merged_model",
        help="Directory to save the final merged model."
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default=None,
        help="HuggingFace Hub username to push the merged model to."
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI."
    )
    return parser.parse_args()

def prepare_data(dataset_path: str) -> DatasetDict:
    """Loads dataset from local JSONL or HuggingFace Hub, splits 90/10, and applies Mistral chat template."""
    logger.info(f"Loading dataset from: {dataset_path}...")
    
    # 1. Load dataset
    if os.path.exists(dataset_path):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        # Load from HF Hub
        dataset = load_dataset(dataset_path, split="train")

    # 2. Train/val split (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 3. Format with Mistral instruction template:
    # <s>[INST] {instruction}\nContext: {context} [/INST] {response}</s>
    def apply_mistral_template(example: Dict[str, Any]) -> Dict[str, Any]:
        instruction = example.get("instruction", "")
        context = example.get("context", "")
        response = example.get("response", "")
        
        formatted_text = f"<s>[INST] {instruction}\nContext: {context} [/INST] {response}</s>"
        return {"text": formatted_text}

    formatted_dataset = split_dataset.map(
        apply_mistral_template,
        remove_columns=dataset.column_names
    )
    
    return DatasetDict({
        "train": formatted_dataset["train"],
        "validation": formatted_dataset["test"]
    })

def run_finetuning(args: argparse.Namespace) -> None:
    # 1. Set MLflow tracking if provided and mlflow is available
    if mlflow is not None:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("idip-llm-finetuning")
    else:
        logger.warning("MLflow is not installed. Experiment tracking is disabled.")

    # 2. Setup Quantization Config (4-bit NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 3. Load Tokenizer & Model in 4-bit
    logger.info(f"Loading quantized base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA parameters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 5. Load and prepare dataset splits
    dataset = prepare_data(args.dataset_path)

    # 6. SFT Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        report_to=["mlflow"] if mlflow is not None else [],
        logging_dir=f"{args.output_dir}/logs"
    )

    # 7. SFTTrainer initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args
    )

    # 8. Start MLflow run & train if available, else standard training run
    logger.info("Starting SFTTrainer fine-tuning...")
    if mlflow is not None:
        with mlflow.start_run() as run:
            # Log training parameters
            mlflow.log_params({
                "base_model": args.base_model,
                "epochs": args.epochs,
                "lora_r": 16,
                "lora_alpha": 32,
                "learning_rate": 2e-4
            })
            
            trainer.train()
            
            # Save trained adapters
            logger.info(f"Saving fine-tuned adapters to: {args.output_dir}...")
            trainer.model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            # Log checkpoints to MLflow
            mlflow.log_artifacts(args.output_dir, artifact_path="lora_adapters")
    else:
        trainer.train()
        # Save trained adapters
        logger.info(f"Saving fine-tuned adapters to: {args.output_dir}...")
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # 9. Merge adapters back into base model
    logger.info("Merging LoRA adapters into base model...")
    merge_adapters(args.base_model, args.output_dir, args.merged_dir)

    # 10. Push merged model to Hugging Face Hub
    if args.hf_username:
        repo_id = f"{args.hf_username}/idip-mistral-finetuned"
        logger.info(f"Pushing merged model to HuggingFace Hub: {repo_id}...")
        try:
            merged_model = AutoModelForCausalLM.from_pretrained(args.merged_dir)
            merged_model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            logger.info("Model pushed successfully.")
        except Exception as e:
            logger.error(f"Failed to push model to HF Hub: {e}")

def merge_adapters(base_model_id: str, adapter_dir: str, output_dir: str) -> None:
    """Loads base model in float16, attaches LoRA adapter, merges, and persists to disk."""
    try:
        # Load base model in 16-bit floating point format (not quantized)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        # Load the PEFT model containing LoRA layers
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        
        # Merge LoRA weights into base parameters
        merged_model = peft_model.merge_and_unload()
        
        # Save to disk
        merged_model.save_pretrained(output_dir)
        logger.info(f"Merged model successfully written to: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to merge model adapters: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arguments = parse_args()
    run_finetuning(arguments)
