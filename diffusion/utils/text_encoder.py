# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os

# === CONFIGURATION ===
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LlamaTextEncoder:
    def __init__(self, model_name=LLAMA_MODEL_NAME, embedding_dim=None, hf_token=None):
        """
        Initialize LLaMA tokenizer and model.
        
        Args:
            model_name (str): HF model name for LLaMA.
            embedding_dim (int): Embedding size (optional). Will infer if None.
            use_auth_token (bool): Whether to use Hugging Face auth token.
        """
        print(f"Initializing Text encoder: {model_name} on {DEVICE}")
        
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        self.model = AutoModel.from_pretrained(model_name, token=hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            print("No pad_token found. Adding pad_token as eos_token...")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Infer embedding dimension if not provided
        self.embedding_dim = embedding_dim or self.model.config.hidden_size
        print(f"Text encoder initialized. Embedding dim: {self.embedding_dim}")

    @torch.no_grad()
    def encode(self, text_list, batch_size=16, pooling="cls"):
        """
        Encode a list of text strings into embeddings.
        
        Args:
            text_list (list of str): List of texts.
            batch_size (int): Batch size for inference.
            pooling (str): 'cls' or 'mean'.
        
        Returns:
            np.ndarray: Embedding array of shape (num_texts, embedding_dim).
        """
        embeddings = []
        print(f"Encoding {len(text_list)} texts...")

        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]

            tokens = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**tokens)

            # Pooling
            if pooling == "cls":
                # Use CLS token from last hidden state
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
            elif pooling == "mean":
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                summed = (outputs.last_hidden_state * attention_mask).sum(1)
                count = attention_mask.sum(1)
                batch_embeddings = summed / count
            else:
                raise ValueError(f"Unsupported pooling type: {pooling}")

            embeddings.append(batch_embeddings.cpu().numpy())

        # Stack all batches
        embeddings = np.vstack(embeddings)
        print(f"Encoding complete! Shape: {embeddings.shape}")
        return embeddings

    def encode_single(self, text, pooling="cls"):
        """
        Encode a single text string.
        """
        return self.encode([text], pooling=pooling)[0]

