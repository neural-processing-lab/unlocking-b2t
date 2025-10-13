import lightning as L
import torch
import torch.nn.functional as F
import bert_score
import jiwer
import tqdm

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix
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

        # Confusion matrix for word predictions
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes)

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
        subjects = batch["subject_id"].squeeze(0)
        sensor_xyz = batch["sensor_xyz"]
        dataset_id = batch["dataset_id"]
        logits = self(x, subjects, sensor_xyz, dataset_id)
        # loss = self._clip_loss(logits, y)
        loss = self._siglip_loss(logits, y, discard=True)
        similarities = self._get_prediction(logits)        

        # Mask invalid words
        valid_mask = y != -1
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

            # Generate predicted transcript using beam search
            pred_sent = qwen_beam_search(
                asr_predictions=full_preds,
                vocabulary=[w.lower() for w in self.top_words_map.keys()],
                beam_width=self.beam_width,
            ).strip()

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
        # Compute and log confusion matrix for word predictions
        true_indices = []
        pred_indices = []
        for ws in self.test_step_outputs:
            for true_word in ws["true_words"]:
                true_indices.append(self.top_words_map[true_word.upper()])
            for pred_word in ws["pred_words"]:
                pred_indices.append(self.top_words_map[pred_word.upper()])

        if len(true_indices) > 0 and len(pred_indices) > 0:
            true_tensor = torch.tensor(true_indices, device=self.device)
            pred_tensor = torch.tensor(pred_indices, device=self.device)

            self.test_confusion_matrix(pred_tensor, true_tensor)
            cm = self.test_confusion_matrix.compute()

            # Log confusion matrix as a figure
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(20, 20))
            sns.heatmap(cm.cpu().numpy(), cmap='Blues', ax=ax, square=True, cbar=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Word Prediction Confusion Matrix')

            self.logger.experiment.log({"confusion_matrix": fig})
            plt.close()

        if self.post_proc:
            results = self.test_step_outputs
            data = []

            print("Logging predictions...")
            for result in tqdm.tqdm(results):
                true_sent = result["true_sent"]

                if not self.greedy_only:
                    beam_sent = result["beam_sent"]
                    self._log_sentence_metrics(true_sent, beam_sent, prefix="beam")
                else:
                    beam_sent = ""

                greedy_sent = result["greedy_sent"]
                self._log_sentence_metrics(true_sent, greedy_sent, prefix="greedy")

                data.append([true_sent, greedy_sent, beam_sent])

            self.logger.log_text(
                key="predictions",
                columns=["true", "greedy", "beam"],
                data=data
            )

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
        top_50_preds = preds[top_50_mask]
        top_50_y = y[top_50_mask]
        self.topk_top50_test_acc(top_50_preds, top_50_y)
        self.log(f"{prefix}_top50_top10acc", self.topk_top50_test_acc)

        # Collect word predictions for confusion matrix and accuracy
        true_words = [self.top_idx_map[x.item()].lower() for x in full_y if x != -1]
        pred_words = [self.top_idx_map[x.item()].lower() for x in preds.argmax(dim=-1)]
        cosine_sim = preds.max(dim=-1).values

        if self.post_proc:
            true = [w[0] for w in batch["words_raw"]]
            true_sent = " ".join(true).lower()

            if not self.greedy_only:
                beam_sent = qwen_beam_search(
                    asr_predictions=full_preds,
                    vocabulary=[w.lower() for w in self.top_words_map.keys()],
                    beam_width=self.beam_width,
                ).strip()
            else:
                beam_sent = ""

            # Compute greedy sentence
            greedy_indices = full_preds.argmax(dim=-1)
            greedy_sent = " ".join([
                self.top_idx_map[x.item()].lower() for x in greedy_indices
            ])

            self.test_step_outputs.append({
                "true_sent": true_sent,
                "beam_sent": beam_sent,
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
        parser.add_argument("--har_type", type=str, default="gating") # default='spatial_attention')
        parser.add_argument("--embedding_dim", type=int, default=1024)
        parser.add_argument("--post_proc", action='store_true', default=False)
        parser.add_argument("--greedy_only", action='store_true', default=False)
        parser.add_argument("--random_noise_inputs", action='store_true', default=False)
        return parent_parser