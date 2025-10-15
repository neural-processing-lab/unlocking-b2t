import lightning as L
import torch
import torch.nn.functional as F
import bert_score
import jiwer
import tqdm
import csv
import os
from datetime import datetime

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy
from x_transformers import Encoder

from model.brainmagick.brain_model import BrainModel
from model.qwen_beam_search import qwen_beam_search
from model.contrastive import SigLipLoss

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


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

        self.beam_width = kwargs["beam_width"]
        self.temperature = kwargs["temperature"]
        self.num_return_sequences = kwargs["num_return_sequences"]
        self.post_proc = kwargs["post_proc"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.greedy_only = kwargs["greedy_only"]
        self.random_noise_inputs = kwargs["random_noise_inputs"]

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
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)

        self.topk_train_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)
        self.topk_val_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)
        self.topk_test_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)
        self.topk_top50_test_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes, top_k=10)

        self.val_beam_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)
        self.test_beam_acc = Accuracy(task="multiclass", average="macro", num_classes=n_classes)

        self.siglip_loss = SigLipLoss()

        self.test_step_outputs = []
        self.predict_step_outputs = []

        # For transcript generation
        self.save_transcripts = False
        self.transcript_output_file = None

        self.register_buffer("word_embeddings", word_embeddings)

    def forward(self, x, subjects, sensor_xyz, dataset_id):
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
        x = x[0, :, :] # Take contextual embeddings [B, dim]
        x = self.projector(x)

        return x
    
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

        # Create mask for valid indices (filter out-of-vocab: -1, and padding: -2)
        valid_mask = (word_indices != -1) & (word_indices != -2)

        # Filter out invalid entries
        valid_brain_features = brain_features[valid_mask]
        valid_word_indices = word_indices[valid_mask]
        
        # Only proceed if we have valid samples
        if valid_brain_features.shape[0] == 0:
            return torch.tensor(0.0, device=brain_features.device)
        
        valid_word_features = self.word_embeddings[valid_word_indices]
      
        return self.siglip_loss(valid_brain_features, valid_word_features, reweigh_positives=discard)
    
    def _clip_loss(self, brain_features, word_indices, temperature=0.07):

        # Create mask for valid indices (filter out-of-vocab: -1, and padding: -2)
        valid_mask = (word_indices != -1) & (word_indices != -2)

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
        subjects = batch["subject_id"].squeeze(0)
        sensor_xyz = batch["sensor_xyz"]
        dataset_id = batch["dataset_id"]
        logits = self(x, subjects, sensor_xyz, dataset_id)
        # loss = self._clip_loss(logits, y)
        loss = self._siglip_loss(logits, y, discard=True)
        similarities = self._get_prediction(logits)

        # Mask invalid words (out-of-vocab: -1, padding: -2)
        valid_mask = (y != -1) & (y != -2)
        valid_y = y[valid_mask]
        valid_similarities = similarities[valid_mask]


        return loss, valid_similarities, valid_y, similarities, y
    
    def predict_step(self, batch, batch_idx):
        loss, preds, y, full_preds, full_y = self._shared_step(batch, "predict")

        # Handle transcript generation if requested
        if self.save_transcripts:
            # Get true sentence
            true_words = [w[0] for w in batch["words_raw"]]
            true_sent = " ".join(true_words).lower()

            # Get sentence length if available (for sentence-aligned mode)
            sentence_length = batch.get("sentence_length")
            if sentence_length is not None:
                sentence_length = sentence_length.item()
                # Slice predictions to actual sentence length
                sentence_preds = full_preds[:sentence_length]
            else:
                sentence_preds = full_preds
                sentence_length = None

            # Generate predicted transcript using beam search (get top beam)
            pred_sents = qwen_beam_search(
                asr_predictions=sentence_preds,
                vocabulary=[w.lower() for w in self.top_words_map.keys()],
                beam_width=self.beam_width,
                num_return_sequences=1,
                temperature=self.temperature,
                max_length=sentence_length,
            )
            pred_sent = pred_sents[0].strip()

            # Store predictions for later logging
            self.predict_step_outputs.append({
                "transcript": pred_sent,
                "target": true_sent,
            })
            return



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



    def _compute_sentence_metrics(self, true_sent, pred_sent):
        cer = jiwer.cer(true_sent, pred_sent)
        wer = jiwer.wer(true_sent, pred_sent)

        ref_tokens = true_sent.split()
        hyp_tokens = pred_sent.split()

        bleu = sentence_bleu([ref_tokens], hyp_tokens, weights=[1]) # BLEU-1 only

        rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_score = rouge.score(true_sent, pred_sent)

        bert = bert_score.score([pred_sent], [true_sent], lang='en', verbose=False)

        return {
            "cer": cer,
            "wer": wer,
            "bleu": bleu,
            "rouge": rouge_score['rouge1'].fmeasure,
            "bert": bert[2].item(),
        }

    def _log_sentence_metrics(self, true_sent, pred_sent, prefix="pred"):
        metrics = self._compute_sentence_metrics(true_sent, pred_sent)

        cer = metrics["cer"]
        wer = metrics["wer"]
        bleu = metrics["bleu"]
        rouge_score = metrics["rouge"]
        bert = metrics["bert"]

        self.log(f"test_{prefix}_cer", cer)
        self.log(f"test_{prefix}_wer", wer, prog_bar=True)
        self.log(f"test_{prefix}_bleu", bleu)
        self.log(f"test_{prefix}_rouge", rouge_score)
        self.log(f"test_{prefix}_bert", bert)


    def on_test_epoch_end(self):
        if self.post_proc:
            results = self.test_step_outputs
            data = []
            csv_data = []

            # Print first example's predictions for inspection
            if results and not self.greedy_only:
                first_result = results[0]
                print("\n" + "="*80)
                print("FIRST TEST EXAMPLE - Top-K Beam Predictions:")
                print("="*80)
                print(f"TRUE:   {first_result['true_sent']}")
                print(f"GREEDY: {first_result['greedy_sent']}")
                print(f"\nTop-{len(first_result['beam_sents'])} Beam Predictions:")
                for i, beam_sent in enumerate(first_result['beam_sents'], 1):
                    print(f"  [{i}] {beam_sent}")
                print("="*80 + "\n")

            print("Logging predictions...")
            for result in tqdm.tqdm(results):
                true_sent = result["true_sent"]
                greedy_sent = result["greedy_sent"]

                if not self.greedy_only:
                    beam_sents = result["beam_sents"]

                    # Log metrics for the best (first) beam
                    if beam_sents:
                        self._log_sentence_metrics(true_sent, beam_sents[0], prefix="beam_top1")

                    # Format all beam sentences for display
                    beam_display = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(beam_sents)])

                    # Get top beam for CSV (or empty string if no beams)
                    top_beam = beam_sents[0] if beam_sents else ""
                else:
                    beam_display = ""
                    top_beam = ""

                self._log_sentence_metrics(true_sent, greedy_sent, prefix="greedy")

                # Add to table with all beams displayed
                data.append([true_sent, greedy_sent, beam_display])

                # Add to CSV data (only true, greedy, and top beam)
                csv_data.append({
                    "true_sentence": true_sent,
                    "greedy_prediction": greedy_sent,
                    "top_beam_prediction": top_beam
                })

            # Create columns list dynamically
            columns = ["true", "greedy", "beams"]

            self.logger.log_text(
                key="predictions",
                columns=columns,
                data=data
            )

            # Save to CSV file
            os.makedirs("predictions", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"predictions/test_predictions_{timestamp}.csv"

            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["true_sentence", "greedy_prediction", "top_beam_prediction"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

            print(f"\nPredictions saved to {csv_filename}")

        return


    def test_step(self, batch, batch_idx, dataloader_idx=0):

        loss, preds, y, full_preds, full_y = self._shared_step(batch, "test")

        prefix = "test" if dataloader_idx == 0 else "holdout"

        self.test_acc(preds, y)
        self.topk_test_acc(preds, y)
        
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_acc", self.test_acc)
        self.log(f"{prefix}_top10acc", self.topk_test_acc)

        # Also compute top-10 accuracy on the top-50 words only
        top_50_mask = y < 50
        if top_50_mask.any():
            top_50_preds = preds[top_50_mask]
            top_50_y = y[top_50_mask]
            self.topk_top50_test_acc(top_50_preds, top_50_y)
            self.log(f"{prefix}_top50_top10acc", self.topk_top50_test_acc)

        # Collect word predictions for confusion matrix and accuracy
        # Filter out both out-of-vocab (-1) and padding (-2) tokens
        true_words = [self.top_idx_map[x.item()].lower() for x in full_y if x.item() >= 0]
        pred_words = [self.top_idx_map[x.item()].lower() for x in preds.argmax(dim=-1)]
        cosine_sim = preds.max(dim=-1).values

        if self.post_proc:
            # Get sentence length if available (for sentence-aligned test mode)
            sentence_length = batch.get("sentence_length")
            if sentence_length is not None:
                sentence_length = sentence_length.item()
                # Only use words up to sentence length
                true = [w[0] for w in batch["words_raw"][:sentence_length]]
            else:
                true = [w[0] for w in batch["words_raw"]]
            true_sent = " ".join(true).lower()

            # Slice predictions to actual sentence length (already got sentence_length above)
            if sentence_length is not None:
                sentence_preds = full_preds[:sentence_length]
            else:
                sentence_preds = full_preds

            if not self.greedy_only:
                # Get multiple diverse beam sequences
                beam_sents = qwen_beam_search(
                    asr_predictions=sentence_preds,
                    vocabulary=[w.lower() for w in self.top_words_map.keys()],
                    beam_width=self.beam_width,
                    num_return_sequences=self.num_return_sequences,
                    temperature=self.temperature,
                    max_length=sentence_length,
                )
                # Strip whitespace from all returned sequences
                beam_sents = [s.strip() for s in beam_sents]
            else:
                beam_sents = []

            # Compute greedy sentence (use sentence_preds for sentence-aligned mode)
            greedy_indices = sentence_preds.argmax(dim=-1)
            greedy_sent = " ".join([
                self.top_idx_map[x.item()].lower() for x in greedy_indices
            ])

            self.test_step_outputs.append({
                "true_sent": true_sent,
                "beam_sents": beam_sents,  # Now a list of sentences
                "greedy_sent": greedy_sent,
                "true_words": true_words,
                "pred_words": pred_words,
                "cosine_sim": cosine_sim,
            })
        else:
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
        parser.add_argument("--beam_width", type=int, default=5)
        parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for beam search diversity (>1 = more diverse)")
        parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of diverse beam sequences to return")
        parser.add_argument("--har_type", type=str, default="gating") # default='spatial_attention')
        parser.add_argument("--embedding_dim", type=int, default=1024)
        parser.add_argument("--post_proc", action='store_true', default=False)
        parser.add_argument("--greedy_only", action='store_true', default=False)
        parser.add_argument("--random_noise_inputs", action='store_true', default=False)
        return parent_parser