import asyncio
import lightning as L
import random
import torch
import torch.nn.functional as F
import bert_score
import jiwer
import tqdm
import pandas as pd
import os

from collections import defaultdict

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, AUROC
from x_transformers import Encoder
from peft import LoraConfig, TaskType, get_peft_model
from transformers import T5ForConditionalGeneration, T5Tokenizer

from model.brainmagick.brain_model import BrainModel
from model.contrastive import SigLipLoss
from model import oov_predictor

import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

import textdistance
from g2p_en import G2p
g2p = G2p()

# Download required NLTK resources (only needed once)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng') # For g2p_en

def top_n_word_error_rate(reference, candidates):
    """
    Calculate top-n word accuracy rate.
    
    Args:
        reference: string with space-separated words
        candidates: list of n candidate strings
    
    Returns:
        float: WER as 1 - hits/total_words
    """
    ref_words = reference.split()
    candidate_words = [candidate.split() for candidate in candidates]
    
    hits = 0
    total_words = len(ref_words)
    
    for pos in range(total_words):
        ref_word = ref_words[pos]
        # Check if ANY candidate has the correct word at this position
        if any(candidate_words[i][pos] == ref_word for i in range(len(candidates))):
            hits += 1
    
    return 1 - hits / total_words


class WordClassifier(L.LightningModule):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        word_embeddings: torch.Tensor,
        top_words_map: dict,
        other_words: list,
        n_subjects: int,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.top_words_map = top_words_map
        self.top_idx_map = {v: k for k, v in top_words_map.items()}
        self.other_words = set(other_words)

        self.embedding_dim = kwargs["embedding_dim"]
        self.limit_context = kwargs["limit_context"]
        self.random_noise_inputs = kwargs["random_noise_inputs"]
        self.temperature = kwargs["temperature"]
        self.top_p = kwargs["top_p"]
        self.n_gens = kwargs["n_gens"]

        self.model = BrainModel(
            in_channels=n_channels,
            out_channels=1024,
            n_subjects=n_subjects,
            dataset=kwargs["dataset"],
            har_type=kwargs["har_type"],
        )
  
        self.transformer = Encoder(
            dim = 1024,  # must match your embedding dimension
            depth = 16, #8,
            heads = 16, #8,
            rotary_pos_emb = True,
            attn_dropout = 0.1
        )
        self.projector = torch.nn.Linear(1024, self.embedding_dim)
        
        # T5 model for sequence generation
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
        
        # LoRA configuration for T5
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,  # Low rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]  # T5 attention and FFN modules
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        
        # Project brain embeddings to T5 hidden size (1024 for t5-large)
        self.t5_projector = torch.nn.Linear(1024, self.t5_model.config.d_model)
        
        # Loss weight for T5 generation
        self.t5_loss_weight = kwargs.get("t5_loss_weight", 1.0)
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)

        self.topk_train_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)
        self.topk_val_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)
        self.topk_test_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)
        self.topk_top50_test_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)

        self.siglip_loss = SigLipLoss()

        self.test_step_outputs = []
        self.predict_step_outputs = []

        self.register_buffer("word_embeddings", word_embeddings)

    def forward(self, x, subjects, sensor_xyz, dataset_id, target_tokens=None, return_t5_output=False):
        # (B, C, T)

        if self.random_noise_inputs:
            # Construct random noise input of same shape as x and same mean and std
            noise = torch.randn_like(x)
            noise_mean = x.mean(dim=-1, keepdim=True)
            noise_std = x.std(dim=-1, keepdim=True)
            x = noise_mean + noise_std * noise

        x = self.model(x, subjects, sensor_xyz, dataset_id) # (B, E, T)
        x = x.mean(dim=-1) # average over time (B, 1024)
        x = x.unsqueeze(0) # (1, B, 1024)

        x = self.transformer(x)
        x = x[0, :, :] # Take contextual embeddings [B, dim] -> [64, embedding_dim]
        x = self.projector(x) # [64, embedding_dim]

        if not return_t5_output:
            return x
        
        # T5 path: x is [64, embedding_dim], add batch dimension for T5
        x_t5 = x.unsqueeze(0)  # [1, 64, embedding_dim]
        x_t5 = self.t5_projector(x_t5)  # Project to T5 hidden size [1, 64, d_model]
        
        # Generate with T5 by injecting at layer 12
        t5_output = self._t5_generate_with_injection(x_t5, target_tokens)
        
        return x, t5_output
    
    def _t5_generate_with_injection(self, brain_embeddings, target_tokens):
        """
        Generate with T5 by providing brain embeddings directly as encoder outputs.
        
        Args:
            brain_embeddings: [1, 64, d_model] brain-derived embeddings
            target_tokens: [1, seq_len] target token ids for training
            
        Returns:
            T5 output logits or loss depending on whether target_tokens is provided
        """
        from transformers.modeling_outputs import BaseModelOutput
        
        # Create encoder outputs object with our brain embeddings
        encoder_outputs = BaseModelOutput(last_hidden_state=brain_embeddings)
        
        batch_size = brain_embeddings.shape[0]
        
        if target_tokens is not None:
            # Training mode: use teacher forcing
            # print(f"Target tokens shape: {target_tokens.shape}, dtype: {target_tokens.dtype}")
            # print(f"Target tokens device: {target_tokens.device}")
            # print(f"Brain embeddings device: {brain_embeddings.device}")
            
            # Ensure target tokens are on the right device and of correct type
            target_tokens = target_tokens.to(brain_embeddings.device).long()
            
            # Create decoder input by shifting target tokens
            decoder_input_ids = torch.zeros_like(target_tokens)
            decoder_input_ids[:, 1:] = target_tokens[:, :-1]
            decoder_input_ids[:, 0] = self.t5_tokenizer.pad_token_id
            
            # print(f"Decoder input shape: {decoder_input_ids.shape}")
            # print(f"Decoder input sample: {decoder_input_ids[0, :10]}")
            
            # Run T5 decoder with brain embeddings as encoder outputs
            outputs = self.t5_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=target_tokens,
                return_dict=True
            )
            return outputs
        else:

            gens = []
            for _ in range(self.n_gens):
                # Inference mode: generate tokens
                # Start with pad token
                generated_ids = torch.full(
                    (batch_size, 1), 
                    self.t5_tokenizer.pad_token_id, 
                    device=brain_embeddings.device
                )
                
                max_length = 120  # Maximum sequence length for generation.
                
                for _ in range(max_length):
                    outputs = self.t5_model(
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=generated_ids,
                        return_dict=True
                    )
                    
                    next_token_logits = outputs.logits[:, -1, :] / self.temperature  # Apply temperature
                    
                    # Apply top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Create mask for tokens to keep (cumulative probability <= top_p)
                    sorted_indices_to_remove = cumulative_probs > self.top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Stop if EOS token is generated
                    if next_token.item() == self.t5_tokenizer.eos_token_id:
                        break
                
                gens.append(generated_ids)
                    
            return gens
    
    def _siglip_loss(self, brain_features, word_indices, discard=False):
        """
        SigLIP loss function.
        
        Args:
            embeddings1: First set of embeddings (e.g., brain features)
            embeddings2: Second set of embeddings (e.g., word features)
            temperature: Scaling factor for logits
            bias: Optional bias term
            
        Returns:
            SigLIP loss value
        """

        # Create mask for valid indices
        valid_mask = word_indices != -1
        
        # Filter out invalid entries
        valid_brain_features = brain_features[valid_mask]
        valid_word_indices = word_indices[valid_mask]
        
        # Only proceed if we have valid samples
        if valid_brain_features.shape[0] == 0:
            return torch.tensor(0.0, device=brain_features.device)
        
        valid_word_features = self.word_embeddings[valid_word_indices]
      
        return self.siglip_loss(valid_brain_features, valid_word_features, reweigh_positives=discard)
    
    def _clip_loss(self, brain_features, word_indices, temperature=0.07):

        # Create mask for valid indices
        valid_mask = word_indices != -1
        
        # Filter out invalid entries
        valid_brain_features = brain_features[valid_mask]
        valid_word_indices = word_indices[valid_mask]
        
        # Only proceed if we have valid samples
        if valid_brain_features.shape[0] == 0:
            return torch.tensor(0.0, device=brain_features.device)
        
        word_features = self.word_embeddings[valid_word_indices]

        valid_brain_features = F.normalize(valid_brain_features, dim=-1)
        word_features = F.normalize(word_features, dim=-1)

        similarity = valid_brain_features @ word_features.T / temperature

        # Labels are the diagonal elements for valid samples only
        labels = torch.arange(valid_brain_features.shape[0], device=brain_features.device)
        
        loss_brain = F.cross_entropy(similarity, labels)
        loss_word = F.cross_entropy(similarity.T, labels)

        total_loss = (loss_brain + loss_word) / 2
        
        return total_loss
    
    def _get_prediction(self, brain_features):
        brain_features = F.normalize(brain_features, dim=-1)
        word_features = F.normalize(self.word_embeddings, dim=-1)
        similarity = brain_features @ word_features.T
        return similarity
    
    def _shared_step(self, batch, stage):
        x = batch["meg"].squeeze(0)
        y = batch["words"].squeeze(0)
        y_tokenized = batch["words_tokenized"].squeeze(0)
        subjects = batch["subject_id"].squeeze(0)
        sensor_xyz = batch["sensor_xyz"]
        dataset_id = batch["dataset_id"]
        
        # Debug tokenized input
        # print(f"y_tokenized shape: {y_tokenized.shape}, dtype: {y_tokenized.dtype}")
        # print(f"y_tokenized sample: {y_tokenized[:10]}")
        
        # Get both SigLIP embeddings and T5 output
        siglip_logits, t5_output = self(x, subjects, sensor_xyz, dataset_id, 
                                      target_tokens=y_tokenized.unsqueeze(0), 
                                      return_t5_output=True)
        
        # SigLIP loss (existing)
        siglip_loss = self._siglip_loss(siglip_logits, y, discard=True)

        
        # T5 generation loss
        t5_loss = t5_output.loss if hasattr(t5_output, 'loss') else torch.tensor(0.0, device=x.device)
        print(f"{stage} SigLIP loss: {siglip_loss.item()}, T5 loss: {t5_loss.item()}")
        
        # Debug T5 output
        # if hasattr(t5_output, 'logits'):
            # print(f"T5 logits shape: {t5_output.logits.shape}")
            # print(f"T5 logits contains NaN: {torch.isnan(t5_output.logits).any()}")
        
        # Combined loss
        total_loss = siglip_loss + self.t5_loss_weight * t5_loss
        
        similarities = self._get_prediction(siglip_logits)        

        # Mask invalid words
        valid_mask = y != -1
        valid_y = y[valid_mask]
        valid_similarities = similarities[valid_mask]

        return total_loss, valid_similarities, valid_y, similarities, y


    def training_step(self, batch, batch_idx):
        loss, preds, y, full_preds, full_y = self._shared_step(batch, "train")

        # Skip top-10 accuracy calculation if there are less than 10 valid words
        if y.shape[0] > 1:
            self.train_acc(preds, y)
            self.topk_train_acc(preds, y)
            self.log("train_top10acc", self.topk_train_acc, prog_bar=True)
            self.log("train_acc", self.train_acc, prog_bar=True)
        else:
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        self.log("train_loss", loss)

        return loss
        

    def validation_step(self, batch, batch_idx):
        loss, preds, y, full_preds, full_y = self._shared_step(batch, "val")
        self.val_acc(preds, y)
        self.topk_val_acc(preds, y)
        
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_top10acc", self.topk_val_acc, prog_bar=True)
        self.log("val_loss", loss)
        
        # Generate T5 output for validation monitoring
        x = batch["meg"].squeeze(0)
        y_tokenized = batch["words_tokenized"].squeeze(0)
        subjects = batch["subject_id"].squeeze(0)
        sensor_xyz = batch["sensor_xyz"]
        dataset_id = batch["dataset_id"]
        
        # Get T5 generation for inference (without teacher forcing)
        _, t5_generated = self(x, subjects, sensor_xyz, dataset_id, 
                             target_tokens=None, 
                             return_t5_output=True)
        
        # T5 WER evaluation
        if t5_generated is not None:

            generated_texts = []
            for gen in t5_generated:
                # Decode T5 generated tokens
                generated_text = self.t5_tokenizer.decode(gen.squeeze(), skip_special_tokens=True)

                # Truncate at 64 words
                generated_text = " ".join(generated_text.split()[:64])

                generated_texts.append(generated_text)
            
            # Create ground truth text from tokenized input
            true_text = self.t5_tokenizer.decode(y_tokenized, skip_special_tokens=True)
            
            # Print T5 outputs for inspection
            for q, generated_text in enumerate(generated_texts):
                print(f"Val T5 Generated {q}: '{generated_text}'")
            print(f"Val Ground Truth: '{true_text}'")

            # Compute top n wer
            top_n_wer = top_n_word_error_rate(reference=true_text, candidates=generated_texts)
            self.log(f"val_top_{self.n_gens}_wer", top_n_wer)
            print(f"Val Top-{self.n_gens} WER: {top_n_wer:.4f}")

            # # Compute T5 WER
            # t5_wer = jiwer.wer(true_text.lower(), generated_text.lower())
            # self.log("val_t5_wer", t5_wer)
            # print(f"Val T5 WER: {t5_wer:.4f}")
            print("-" * 50)

    def _compute_sentence_metrics(self, true_sent, pred_sent):

        cer = jiwer.cer(true_sent, pred_sent)

        jiwer_output = jiwer.process_words(true_sent, pred_sent)
        wer = jiwer_output.wer
        wil = jiwer_output.wil
        wip = jiwer_output.wip
        mer = jiwer_output.mer

        ref_tokens = true_sent.split()
        hyp_tokens = pred_sent.split()

        # Convert words to phonemes
        true_words = true_sent.split()
        pred_words = pred_sent.split()
        true_tokens = [g2p(word)[:2] for word in true_words]
        pred_tokens = [g2p(word)[:2] for word in pred_words]
        true_tokens = [item for sublist in true_tokens for item in sublist]
        pred_tokens = [item for sublist in pred_tokens for item in sublist]
        per = textdistance.levenshtein(true_tokens, pred_tokens)


        bleu = sentence_bleu([ref_tokens], hyp_tokens, weights=[1]) # BLEU-1 only
        meteor = meteor_score([ref_tokens], hyp_tokens)

        rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_score = rouge.score(true_sent, pred_sent)

        bert = bert_score.score([pred_sent], [true_sent], lang='en', verbose=False)

        return {
            "cer": cer,
            "wer": wer,
            "wil": wil,
            "wip": wip,
            "mer": mer,
            "bleu": bleu,
            "meteor": meteor,
            "rouge": rouge_score['rouge1'].fmeasure,
            "bert": bert[2].item(),
            # "gemini_emb": gemini_sim.item(),
            "per": per,
        }

    def _log_sentence_metrics(self, true_sent, pred_sent, prefix="pred"):
        
        metrics = self._compute_sentence_metrics(true_sent, pred_sent)

        cer = metrics["cer"]
        wer = metrics["wer"]
        wil = metrics["wil"]
        wip = metrics["wip"]
        mer = metrics["mer"]
        bleu = metrics["bleu"]
        meteor = metrics["meteor"]
        rouge_score = metrics["rouge"]
        bert = metrics["bert"]
        per = metrics["per"]

        self.log(f"test_{prefix}_cer", cer)
        self.log(f"test_{prefix}_wer", wer, prog_bar=True)
        self.log(f"test_{prefix}_wil", wil)
        self.log(f"test_{prefix}_wip", wip)
        self.log(f"test_{prefix}_mer", mer)
        self.log(f"test_{prefix}_bleu", bleu)
        self.log(f"test_{prefix}_meteor", meteor)
        self.log(f"test_{prefix}_rouge", rouge_score)
        self.log(f"test_{prefix}_bert", bert)
        self.log(f"test_{prefix}_per", per)


    def on_test_epoch_end(self):

        return


    def test_step(self, batch, batch_idx, dataloader_idx=0):

        # Get both SigLIP and T5 outputs for evaluation
        x = batch["meg"].squeeze(0)
        y = batch["words"].squeeze(0)
        y_tokenized = batch["words_tokenized"].squeeze(0)
        subjects = batch["subject_id"].squeeze(0)
        sensor_xyz = batch["sensor_xyz"]
        dataset_id = batch["dataset_id"]
        
        # Get SigLIP embeddings and T5 output
        siglip_logits, t5_output = self(x, subjects, sensor_xyz, dataset_id, 
                                      target_tokens=y_tokenized.unsqueeze(0), 
                                      return_t5_output=True)
        
        # Also get T5 generation for inference (without teacher forcing)
        _, t5_generated = self(x, subjects, sensor_xyz, dataset_id, 
                             target_tokens=None, 
                             return_t5_output=True)
        
        # Calculate losses
        siglip_loss = self._siglip_loss(siglip_logits, y, discard=True)
        t5_loss = t5_output.loss if hasattr(t5_output, 'loss') else torch.tensor(0.0, device=x.device)
        total_loss = siglip_loss + self.t5_loss_weight * t5_loss
        
        similarities = self._get_prediction(siglip_logits)
        
        # Mask invalid words for SigLIP evaluation
        valid_mask = y != -1
        valid_y = y[valid_mask]
        valid_similarities = similarities[valid_mask]

        prefix = "test" if dataloader_idx == 0 else "holdout"

        self.test_acc(valid_similarities, valid_y)
        self.topk_test_acc(valid_similarities, valid_y)
        
        self.log(f"{prefix}_loss", total_loss)
        self.log(f"{prefix}_siglip_loss", siglip_loss)
        self.log(f"{prefix}_t5_loss", t5_loss)
        self.log(f"{prefix}_acc", self.test_acc)
        self.log(f"{prefix}_top10acc", self.topk_test_acc)

        # Also compute top-10 accuracy on the top-50 words only
        top_50_mask = valid_y < 50
        top_50_preds = valid_similarities[top_50_mask]
        top_50_y = valid_y[top_50_mask]
        self.topk_top50_test_acc(top_50_preds, top_50_y)
        self.log(f"{prefix}_top50_top10acc", self.topk_top50_test_acc)

        # T5 WER evaluation
        if t5_generated is not None:

            generated_texts = []
            for gen in t5_generated:
                # Decode T5 generated tokens
                generated_text = self.t5_tokenizer.decode(gen.squeeze(), skip_special_tokens=True)

                # Truncate at 64 words
                generated_text = " ".join(generated_text.split()[:64])

                generated_texts.append(generated_text)
            
            # Create ground truth text from tokenized input
            true_text = self.t5_tokenizer.decode(y_tokenized, skip_special_tokens=True)
            
            # Print T5 outputs for inspection
            for q, generated_text in enumerate(generated_texts):
                print(f"{prefix} T5 Generated {q}: '{generated_text}'")
            print(f"Ground Truth: '{true_text}'")
            
            # Compute T5 WER
            top_n_wer = top_n_word_error_rate(reference=true_text, candidates=generated_texts)
            self.log(f"{prefix}_top{self.n_gens}_t5_wer", top_n_wer)
            print(f"T5 top-{self.n_gens} WER: {top_n_wer:.4f}")
            print("-" * 50)

        # Log greedy and true words that are within vocabulary (SigLIP)
        true_words = [self.top_idx_map[x.item()].lower() for x in y if x != -1]
        pred_words = [self.top_idx_map[x.item()].lower() for x in valid_similarities.argmax(dim=-1)]
        cosine_sim = valid_similarities.max(dim=-1).values

        self.test_step_outputs.append({
            "true_words": true_words,
            "pred_words": pred_words,
            "cosine_sim": cosine_sim,
        })

        return


    def configure_optimizers(self):
        # Use AdamW and filter parameters for those that require grad
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,  # decay over 50 epochs
            eta_min=1e-6  # minimum learning rate
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WordClassifier")
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--har_type", type=str, default='spatial_attention')
        parser.add_argument("--embedding_dim", type=int, default=1024)
        parser.add_argument("--limit_context", type=int, default=8)
        parser.add_argument("--random_noise_inputs", action='store_true', default=False)
        parser.add_argument("--t5_loss_weight", type=float, default=1.0)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--n_gens", type=int, default=10, help="Number of generations for T5")
        return parent_parser