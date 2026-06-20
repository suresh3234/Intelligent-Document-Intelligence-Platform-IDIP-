import json
import pytest
import numpy as np
import asyncio
import torch
from unittest.mock import MagicMock, patch, mock_open
from models.llm.eval import LLMEvaluator
from models.llm.finetune import prepare_data, run_finetuning
from models.llm.inference import (
    LLMInferenceService,
    latency_ms_gauge,
    tokens_per_second_gauge,
    total_tokens_counter
)

# --- 1. LLM Evaluator Unit Tests ---

def test_evaluator_rouge_l():
    evaluator = LLMEvaluator()
    
    # Exact match
    s1 = "This is a simple sentence."
    assert evaluator.compute_rouge_l(s1, s1) == 1.0
    
    # Disjoint match
    s2 = "Cooking cake."
    assert evaluator.compute_rouge_l(s1, s2) == 0.0
    
    # Empty string check
    assert evaluator.compute_rouge_l("", s1) == 0.0
    
    # Partial match
    s3 = "This is another simple sentence."
    # LCS: ["this", "is", "simple", "sentence"] length 4
    # F1 calculation: (2 * (4/5) * (4/6)) / ((4/5) + (4/6)) = (2 * 0.8 * 0.67) / (0.8 + 0.67)
    score = evaluator.compute_rouge_l(s1, s3)
    assert score > 0.5
    assert score < 1.0

def test_evaluator_bertscore_approx():
    evaluator = LLMEvaluator()
    assert evaluator.compute_bertscore_approx("Exact sentence match.", "Exact sentence match.") == 1.0
    assert evaluator.compute_bertscore_approx("disjoint", "words") == 0.0
    assert evaluator.compute_bertscore_approx("", "word") == 0.0

def test_evaluator_hallucination_rate():
    evaluator = LLMEvaluator()
    
    # Mock NLI pipeline returning contradiction for the first claim and entailment for the second
    mock_nli = MagicMock()
    mock_nli.side_effect = [
        [{"label": "CONTRADICTION", "score": 0.88}],
        [{"label": "ENTAILMENT", "score": 0.92}]
    ]
    
    with patch.object(evaluator, "_nli_pipeline", mock_nli):
        context = "Paris is the capital of France. London is in the UK."
        response = "Paris is in Germany. London is in the UK."
        
        # Sentence splitter splits into 2 sentences
        rate = evaluator.compute_hallucination_rate(context, response)
        # 1 contradiction out of 2 sentences -> 0.5
        assert rate == 0.5

# --- 2. Fine-tuning Data Prep Unit Tests ---

@patch("models.llm.finetune.load_dataset")
def test_finetuning_data_prep(mock_load_dataset):
    # Mock HF dataset object
    mock_dataset = MagicMock()
    mock_dataset.column_names = ["instruction", "context", "response"]
    
    # Mock split function returning mock DatasetDict
    mock_split_dataset = MagicMock()
    
    # Mock map function
    def mock_map_impl(map_func, remove_columns=None):
        # Apply the map function to dummy values
        res = map_func({
            "instruction": "Explain quantum computing.",
            "context": "Context details.",
            "response": "Response details."
        })
        # Verify chat template mapping
        assert "[INST] Explain quantum computing.\nContext: Context details. [/INST] Response details.</s>" in res["text"]
        return {"train": ["formatted_1"], "test": ["formatted_2"]}
        
    mock_split_dataset.map.side_effect = mock_map_impl
    mock_dataset.train_test_split.return_value = mock_split_dataset
    mock_load_dataset.return_value = mock_dataset
    
    splits = prepare_data("mock_dataset_path")
    assert "train" in splits
    assert "validation" in splits

# --- 3. Inference Service Unit Tests ---

@patch("models.llm.inference.AutoTokenizer")
@patch("models.llm.inference.AutoModelForCausalLM")
def test_inference_service_generate(mock_model_cls, mock_tokenizer_cls):
    # Setup mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 50256
    mock_tokenizer.decode.return_value = "Answer: This is a generated answer from the model."
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    
    # Setup mock model
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model_cls.from_pretrained.return_value = mock_model
    
    # Reset prometheus metric values
    total_tokens_counter._value.set(0)
    
    service = LLMInferenceService()
    
    # 1. Test single generate
    response = service.generate("Test prompt")
    
    assert response == "Answer: This is a generated answer from the model."
    assert latency_ms_gauge._value.get() >= 0.0
    assert tokens_per_second_gauge._value.get() >= 0.0
    assert total_tokens_counter._value.get() == 2.0  # 2 output tokens: (5 - 3) = 2
    
    # 2. Test batch generate
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]])}
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    mock_tokenizer.decode.side_effect = ["batch 1", "batch 2"]
    
    batch_responses = service.generate_batch(["prompt 1", "prompt 2"])
    
    assert len(batch_responses) == 2
    assert batch_responses[0] == "batch 1"
    assert batch_responses[1] == "batch 2"
    # Added 6 output tokens (3 per batch item)
    assert total_tokens_counter._value.get() == 8.0  # 2 + 6 = 8

@pytest.mark.asyncio
@patch("models.llm.inference.AutoTokenizer")
@patch("models.llm.inference.AutoModelForCausalLM")
async def test_inference_service_stream(mock_model_cls, mock_tokenizer_cls):
    # Setup mock tokenizer & model
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 50256
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model_cls.from_pretrained.return_value = mock_model
    
    # Patch TextIteratorStreamer inside the generate_stream method
    with patch("models.llm.inference.TextIteratorStreamer") as mock_streamer_cls:
        # Mock streamer act as an iterator yielding mock tokens
        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["This", " is", " streamed", " token."])
        mock_streamer_cls.return_value = mock_streamer
        
        service = LLMInferenceService()
        
        # Read from async generator stream
        tokens = []
        async for token in service.generate_stream("Stream prompt"):
            tokens.append(token)
            
        assert "".join(tokens) == "This is streamed token."
        assert latency_ms_gauge._value.get() >= 0.0
        assert tokens_per_second_gauge._value.get() >= 0.0

# --- 4. Fine-tuning Pipeline Mock Unit Tests ---

@patch("models.llm.finetune.prepare_data")
@patch("models.llm.finetune.AutoTokenizer")
@patch("models.llm.finetune.AutoModelForCausalLM")
@patch("models.llm.finetune.prepare_model_for_kbit_training")
@patch("models.llm.finetune.get_peft_model")
@patch("models.llm.finetune.SFTTrainer")
@patch("models.llm.finetune.merge_adapters")
def test_run_finetuning_pipeline(
    mock_merge,
    mock_trainer_cls,
    mock_get_peft,
    mock_prep_kbit,
    mock_model_cls,
    mock_tokenizer_cls,
    mock_prepare_data
):
    # Setup dataset prepare mock
    mock_prepare_data.return_value = {
        "train": MagicMock(),
        "validation": MagicMock()
    }
    
    # Setup mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    
    # Setup mock model
    mock_model = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_get_peft.return_value = mock_model
    
    # Setup mock trainer
    mock_trainer = MagicMock()
    mock_trainer_cls.return_value = mock_trainer
    
    # Mock mlflow start_run context manager and functions
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    
    # Run finetuning with mlflow mock
    with patch("models.llm.finetune.mlflow", mock_mlflow):
        from argparse import Namespace
        args = Namespace(
            dataset_path="dummy_path.jsonl",
            base_model="dummy-base-model",
            epochs=1,
            output_dir="dummy_output",
            merged_dir="dummy_merged",
            hf_username="dummy_user",
            mlflow_tracking_uri="http://localhost:5000"
        )
        
        run_finetuning(args)
        
        # Verify calls
        mock_prepare_data.assert_called_once_with("dummy_path.jsonl")
        mock_tokenizer_cls.from_pretrained.assert_called_with("dummy-base-model")
        assert mock_model_cls.from_pretrained.call_count == 2
        mock_prep_kbit.assert_called_once()
        mock_get_peft.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_merge.assert_called_once_with("dummy-base-model", "dummy_output", "dummy_merged")
        
        # MLflow assertions
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("idip-llm-finetuning")
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_artifacts.assert_called_once()

    # Run finetuning when mlflow is not installed (None)
    mock_prepare_data.reset_mock()
    mock_tokenizer_cls.from_pretrained.reset_mock()
    mock_model_cls.from_pretrained.reset_mock()
    mock_prep_kbit.reset_mock()
    mock_get_peft.reset_mock()
    mock_trainer.train.reset_mock()
    mock_merge.reset_mock()
    
    with patch("models.llm.finetune.mlflow", None):
        args_no_mlflow = Namespace(
            dataset_path="dummy_path.jsonl",
            base_model="dummy-base-model",
            epochs=1,
            output_dir="dummy_output",
            merged_dir="dummy_merged",
            hf_username=None,
            mlflow_tracking_uri=None
        )
        
        run_finetuning(args_no_mlflow)
        
        # Verify that training still succeeds without mlflow
        mock_prepare_data.assert_called_once_with("dummy_path.jsonl")
        mock_tokenizer_cls.from_pretrained.assert_called_with("dummy-base-model")
        assert mock_model_cls.from_pretrained.call_count == 1
        mock_trainer.train.assert_called_once()
        mock_merge.assert_called_once_with("dummy-base-model", "dummy_output", "dummy_merged")

