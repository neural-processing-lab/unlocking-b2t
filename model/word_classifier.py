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
from transformers import WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from peft import LoraConfig, TaskType, get_peft_model

from model.brainmagick.brain_model import BrainModel
from model.beam_search import beam_search_with_llama_threaded
from model.contrastive import SigLipLoss
from model.custom_whisper_forward import custom_forward
from model import oov_predictor

import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

import model.labram_finetune
import timm.models
from timm.models import create_model

import textdistance
from g2p_en import G2p
g2p = G2p()

# Download required NLTK resources (only needed once)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng') # For g2p_en

# from model.claude_infilling import infill_sentences
from model.infilling import infill_sentences, batch_in_context_beam_search, replace_unks_alt
# from model.deepseek_infilling import model_call
from model.gemini_infilling import model_call as gemini_model_call
from model.claude_infilling import model_call


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
        self.lm_weight = kwargs["lm_weight"]
        self.pretrained_transformer = kwargs["pretrained_transformer"]
        self.pretrained_encoder = kwargs["pretrained_encoder"]
        self.post_proc = kwargs["post_proc"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.limit_context = kwargs["limit_context"]
        self.greedy_only = kwargs["greedy_only"]
        self.no_llm_api = kwargs["no_llm_api"]
        self.random_noise_inputs = kwargs["random_noise_inputs"]
        self.use_kenlm = kwargs["use_kenlm"]

        if not self.pretrained_encoder:
            self.model = BrainModel(
                in_channels=n_channels,
                out_channels=1024,
                n_subjects=n_subjects,
                dataset=kwargs["dataset"],
                har_type=kwargs["har_type"],
            )
        else:
            # Load pretrained LaBraM model
            checkpoint = torch.load(
                "/data/engs-pnpl/lina4368/projects/EEGPT/downstream/Modules/LaBraM/labram-base.pth",
                weights_only=False,
            )
            new_checkpoint = {}
            for k,v in checkpoint['model'].items():
                if k.startswith('student.'):
                    new_checkpoint[k[len('student.'):]] = v
            model = create_model("labram_base_patch200_200", 
                                    qkv_bias=False,
                                    rel_pos_bias=True,
                                    num_classes=4,
                                    drop_rate=0.0,
                                    drop_path_rate=0.1,
                                    attn_drop_rate=0.0,
                                    drop_block_rate=None,
                                    use_mean_pooling=True,
                                    init_scale=0.001,
                                    use_rel_pos_bias=True,
                                    use_abs_pos_emb=True,
                                    init_values=0.1,)
            model.load_state_dict(new_checkpoint, strict=False)
            self.model = model
            self.chan_conv = torch.nn.Conv1d(n_channels, 19, kernel_size=1)
            self.enc_projector = torch.nn.Linear(96, 1024)

            # Freeze parameters to start with
            for blk in model.blocks:
                for p in blk.parameters():
                    p.requires_grad = False
            # NOTE: requires projection of channels to 19-dim + padding of input to 1100-len

        if not self.pretrained_transformer:
            self.transformer = Encoder(
                dim = 1024,  # must match your embedding dimension
                depth = 16, #8,
                heads = 16, #8,
                rotary_pos_emb = True,
                attn_dropout = 0.1
            )
            self.projector = torch.nn.Linear(1024, self.embedding_dim)
        else:
            # Load base Whisper model
            model_name = "openai/whisper-large-v2"  # or another size like "base", "small", "medium"

            # Eliminates the check for 3000 sequence length
            WhisperEncoder.forward = custom_forward

            base_model = WhisperModel.from_pretrained(model_name)

            # We need just the encoder part
            whisper_encoder = base_model

            # current_conv = whisper_encoder.model.encoder.conv1
            # input_conv = torch.nn.Conv1d(1024)
            # whisper_encoder.model.encoder.set_input_embeddings()

            # Configure LoRA for the Whisper encoder
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # Whisper encoder acts as a feature extractor
                inference_mode=False,
                r=256,
                lora_alpha=384, 
                lora_dropout=0.05,
                # Whisper uses different attention layer naming
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
            )

            # Apply LoRA adapter
            self.transformer = get_peft_model(whisper_encoder, peft_config)
            self.adapter = torch.nn.Linear(1024, 1280)
            self.projector = torch.nn.Linear(1280, self.embedding_dim)

        # self.linear_oov_classifier = torch.nn.Linear(self.embedding_dim, 1)
        
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

        self.test_oov_acc = Accuracy(task="multiclass", average="macro", num_classes=2)
        self.test_oov_auroc = AUROC(task="binary")

        self.siglip_loss = SigLipLoss()

        self.test_step_outputs = []
        self.predict_step_outputs = []

        self.register_buffer("word_embeddings", word_embeddings)

    def forward(self, x, subjects, sensor_xyz, dataset_id):
        # (B, C, T)

        if self.random_noise_inputs:
            # Construct random noise input of same shape as x and same mean and std
            noise = torch.randn_like(x)
            noise_mean = x.mean(dim=-1, keepdim=True)
            noise_std = x.std(dim=-1, keepdim=True)
            x = noise_mean + noise_std * noise

        if not self.pretrained_encoder:
            x = self.model(x, subjects, sensor_xyz, dataset_id) # (B, E, T)
            x = x.mean(dim=-1) # average over time (B, 1024)
            x = x.unsqueeze(0) # (1, B, 1024)
        else:
            x = self.chan_conv(x) # C=senor-dim --> 19 (B, 19, T)
            x = torch.nn.functional.pad(x, (0, 1100-x.shape[-1])) # Pad to 1100 length
            B, C, T = x.shape
            if T%200!=0: 
                x = x[:,:,0:T-T%200]
                T = T-T%200
            x = x.reshape((B,C,T//200,200))
            x = self.model.forward_features(x, input_chans=[i for i in range(C+1)], return_all_tokens=True)
            # [B, 96, 200]
            x = x.mean(dim=-1) # average over time (B, 96)
            x = x.unsqueeze(0) # (1, B, 96)
            x = self.enc_projector(x) # (1, B, 1024)

        if self.pretrained_transformer:
            x = self.adapter(x) # (1, B, 1280)
            B, S = x.shape[0], x.shape[1]

            # Whisper encoder expects input in the form (1, 1280, B --> 3000)
            x = x.permute(0, 2, 1)

            x = self.transformer.encoder(
                input_features=x,
                return_dict=True
            ).last_hidden_state
            x = x[0, :, :] # Take contextual embeddings [B, dim]
            x = self.projector(x)
        else:
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

        # # oov predictor
        # oov_logits = self.linear_oov_classifier(logits) # [B, 1024] -> [B, 1]
        # oov_probs = torch.sigmoid(oov_logits)
        # oov_labels = (y == -1).float().unsqueeze(-1) # [B, 1]
        # oov_loss = F.binary_cross_entropy_with_logits(oov_logits, oov_labels)
        # loss += oov_loss
        # if stage in ["train", "val", "test"]:
        #     self.log(f"{stage}_oov_loss", oov_loss, prog_bar=True)
        #     self.log(f"{stage}_oov_acc", self.test_oov_acc(oov_probs.squeeze(), oov_labels.squeeze()))
        #     self.log(f"{stage}_oov_auroc", self.test_oov_auroc(oov_probs.squeeze(), oov_labels.squeeze()))

        return loss, valid_similarities, valid_y, similarities, y
    
    def predict_step(self, batch, batch_idx):
        loss, preds, y, full_preds, full_y = self._shared_step(batch, "predict")

        # Write a row to existing CSV with the set of all probabilities in this step
        df = pd.DataFrame({
            "oov": (full_y == -1).cpu().numpy().tolist(),
            "probs": full_preds.softmax(dim=-1).squeeze().cpu().numpy().tolist(),
        })

        if batch_idx == 0:
            # Overwrite the CSV file if it exists or create a new one
            df.to_csv("train_oov_preds.csv", index=False)
        else:
            # Append to the existing CSV file
            df.to_csv("train_oov_preds.csv", mode='a', header=False, index=False)


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

    def _log_sentence_metrics(self, true_sent, pred_sent, prefix="pred", add_to_csv=False):
        
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
        # gemini_emb = metrics["gemini_emb"]
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
        # self.log(f"test_{prefix}_gemini_emb", gemini_emb)
        self.log(f"test_{prefix}_per", per)

        if add_to_csv:
            with open("sentence_beam_metrics.csv", "a") as f:
                f.write(f"{true_sent},{pred_sent},{cer},{wer},{bleu},{meteor},{rouge_score},{bert}\n")


    def on_test_epoch_end(self):

        # Save word predictions to CSV
        true_words = [w for ws in self.test_step_outputs for w in ws["true_words"]]
        pred_words = [w for ws in self.test_step_outputs for w in ws["pred_words"]]
        random_words = [random.choice(list(self.top_words_map.keys())).lower() for _ in range(len(true_words))]
        df = pd.DataFrame({
            "true_word": true_words,
            "pred_word": pred_words,
            "random_word": random_words,
        })
        df.to_csv("word_preds.csv", index=False)

        if self.post_proc:
            results = self.test_step_outputs

            if not self.greedy_only:

                if not self.no_llm_api:
                    print("Generating in-context beam search predictions with LLM...")
                    prediction_infos = [result["prediction_info"] for result in results]
                    llm_sents = asyncio.run(batch_in_context_beam_search(prediction_infos, model_call))

                    print("Generating LLM in-fillings from algorithmically beamed sentences...")
                    beam_maskeds = [result["beam_masked"] for result in results]
                    # Generate info for in-filling beamed sentences
                    infill_infos = []
                    for beam_masked in beam_maskeds:
                        infill_info = ""
                        for i, token in enumerate(beam_masked.split()):
                            infill_info += f"\n[{i}]: {token.lower()}"
                        infill_infos.append(infill_info.strip())
                    llm_filleds = asyncio.run(infill_sentences(infill_infos, model_call))

            data = []

            # Create sentence beam metrics csv header (replace if existing)
            with open("sentence_beam_metrics.csv", "w") as f:
                f.write("true,pred,cer,wer,bleu,meteor,rouge,bert\n")

            print("Logging predictions...")
            for i, result in tqdm.tqdm(enumerate(results)):
                true_sent = result["true_sent"]

                if not self.greedy_only:
                    beam_sent = result["beam_sent"]
                    beam_sent_filled = result["beam_sent_filled"]
                    beam_sent_random_filled = result["beam_sent_random_filled"]

                    self._log_sentence_metrics(true_sent, beam_sent, prefix="beam")
                    self._log_sentence_metrics(true_sent, beam_sent_filled, prefix="beam_filled", add_to_csv=True)
                    self._log_sentence_metrics(true_sent, beam_sent_random_filled, prefix="beam_random_filled")

                    if not self.no_llm_api:
                        llm_sent = llm_sents[i]
                        llm_filled_sent = llm_filleds[i]
                        self._log_sentence_metrics(true_sent, llm_sent, prefix="llm_search_and_filled")
                        self._log_sentence_metrics(true_sent, llm_filled_sent, prefix="llm_filled")
                    else:
                        llm_sent = ""
                        llm_filled_sent = ""

                else:
                    beam_sent = ""
                    beam_sent_filled = ""
                    beam_sent_random_filled = ""
                    llm_sent = ""
                    llm_filled_sent = ""

                greedy_sent = result["greedy_sent"]
                greedy_random_filled = result["greedy_random_filled"]
                random_sent = result["random_sent"]

                self._log_sentence_metrics(true_sent, greedy_sent, prefix="greedy")
                self._log_sentence_metrics(true_sent, greedy_random_filled, prefix="greedy_random_filled")
                self._log_sentence_metrics(true_sent, random_sent, prefix="random")

                data.append([true_sent, random_sent, greedy_sent, greedy_random_filled, beam_sent, beam_sent_filled, beam_sent_random_filled, llm_sent, llm_filled_sent])

            self.logger.log_text(
                key="predictions",
                columns=["true", "random", "greedy", "greedy_random_filled", "beam", "beam_filled", "beam_random_filled", "llm_search_and_filled", "llm_filled"],
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

        # We want to measure: WER, CER, BLEU-1, ROUGE-1, METEOR, and BERTScore (maybe also NV-Embed similarity)

        # Save top-5 predictions with their similarity to a CSV
        # Something of the format (ground truth, pred1, prob1, pred2, prob2, ...)
        top10 = preds.topk(10, dim=-1)
        top10_preds = top10.indices
        top10_probs = top10.values
        ground_truths = [self.top_idx_map[x.item()].lower() for x in y if x != -1]
        df = pd.DataFrame({
            "ground_truth": ground_truths,
            "pred1": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 0] if x != -1],
            "prob1": [x.item() for x in top10_probs[:, 0] if x != -1],
            "pred2": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 1] if x != -1],
            "prob2": [x.item() for x in top10_probs[:, 1] if x != -1],
            "pred3": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 2] if x != -1],
            "prob3": [x.item() for x in top10_probs[:, 2] if x != -1],
            "pred4": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 3] if x != -1],
            "prob4": [x.item() for x in top10_probs[:, 3] if x != -1],
            "pred5": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 4] if x != -1],
            "prob5": [x.item() for x in top10_probs[:, 4] if x != -1],
            "pred6": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 5] if x != -1],
            "prob6": [x.item() for x in top10_probs[:, 5] if x != -1],
            "pred7": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 6] if x != -1],
            "prob7": [x.item() for x in top10_probs[:, 6] if x != -1],
            "pred8": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 7] if x != -1],
            "prob8": [x.item() for x in top10_probs[:, 7] if x != -1],
            "pred9": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 8] if x != -1],
            "prob9": [x.item() for x in top10_probs[:, 8] if x != -1],
            "pred10": [self.top_idx_map[x.item()].lower() for x in top10_preds[:, 9] if x != -1],
            "prob10": [x.item() for x in top10_probs[:, 9] if x != -1],
        })

        if os.path.exists("top10_preds.csv"):
            df.to_csv("top10_preds.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("top10_preds.csv", index=False)

        # Log greedy and true words that are within vocabulary
        true_words = [self.top_idx_map[x.item()].lower() for x in full_y if x != -1]
        pred_words = [self.top_idx_map[x.item()].lower() for x in preds.argmax(dim=-1)]
        cosine_sim = preds.max(dim=-1).values

        if self.post_proc:

            if hasattr(self, "oov_model"):
                # Predict OOV words with OOV model
                oov_df = pd.DataFrame(full_preds.softmax(dim=-1).squeeze().cpu().numpy().tolist())
                oov_df.columns = [f'prob_{i}' for i in range(oov_df.shape[1])]
                other_df = pd.DataFrame({
                    "oov": (full_y == -1).cpu().numpy().tolist()
                })
                oov_df = pd.concat([oov_df, other_df], axis=1)

                X = oov_df.drop('oov', axis=1)
                y = oov_df['oov']

                X = oov_predictor.add_distribution_features(X)

                X_scaled = self.oov_scaler.transform(X)

                y_pred = self.oov_model.predict(X_scaled)
                y_prob = self.oov_model.predict_proba(X_scaled)[:, 1]
                missing_mask = y_pred == 1

                y_pred = torch.tensor(y_pred).to(self.device)
                y_prob = torch.tensor(y_prob).to(self.device)
                y = torch.tensor(y).to(self.device)

                self.test_oov_acc(y_pred, y)
                self.test_oov_auroc(y_prob, y)
                self.log("test_oov_acc", self.test_oov_acc)
                self.log("test_oov_auroc", self.test_oov_auroc)
            else:
                missing_mask = full_y == -1

            true = [w[0] for w in batch["words_raw"]]
            true_sent = " ".join(true).lower()

            if not self.greedy_only:

                # Get top-5 predictions for every position in the sequence
                topk = full_preds.topk(5, dim=-1)
                topk_preds = topk.indices
                topk_probs = (topk.values / 0.01).softmax(dim=-1) # Sharpen with temperature 0.01
                topk_words = []
                for i in range(topk_preds.shape[0]):
                    if full_y[i] == -1:
                        topk_words.append([])
                    else:
                        topk_words.append([self.top_idx_map[x.item()].lower() for x in topk_preds[i]])
                prediction_info = ""
                for pos in range(len(topk_words)):
                    if len(topk_words[pos]) > 0:
                        line = f"[{pos}]: "
                        for w, p in zip(topk_words[pos], topk_probs[pos]):
                            line += f"({w.lower()}, {p.item():.2f}) "
                        prediction_info += "\n" + line.strip()
                    else:
                        prediction_info += f"\n[{pos}]: [UNK]"

                beam_masked = beam_search_with_llama_threaded(
                    asr_predictions=full_preds,
                    missing_mask=missing_mask,
                    predict_missing=False,
                    vocabulary=[w.lower() for w in self.top_words_map.keys()],
                    beam_width=self.beam_width,
                    lm_weight=self.lm_weight,
                    num_threads=min(self.beam_width, 5),
                    limit_context=self.limit_context,
                    use_kenlm=self.use_kenlm,
                ).strip()
                beam_sent = beam_masked
                # beam_sent = beam_masked.replace("[UNK] ", "").replace("[UNK]", "").strip()

                # Randomly fill [UNK] tokens
                beam_sent_random_filled = replace_unks_alt(beam_masked, list(self.other_words))

                beam_sent_filled = beam_search_with_llama_threaded(
                    asr_predictions=full_preds,
                    missing_mask=missing_mask,
                    predict_missing=True,
                    vocabulary=[w.lower() for w in self.top_words_map.keys()],
                    beam_width=self.beam_width,
                    lm_weight=self.lm_weight,
                    num_threads=min(self.beam_width, 5),
                    limit_context=self.limit_context,
                    use_kenlm=self.use_kenlm,
                )
            else:
                prediction_info = ""
                beam_masked = ""
                beam_sent = ""
                beam_sent_filled = ""
                beam_sent_random_filled = ""
                beam_masked = ""

            # Compute greedy sentence with [UNK] insertions
            greedy_indices = full_preds.argmax(dim=-1)
            greedy_sent = " ".join([
                self.top_idx_map[x.item()].lower() if not missing_mask[i] else "[UNK]" for i, x in enumerate(greedy_indices)
            ])

            # Replace [UNK]s with random words outside vocabulary
            greedy_random_filled = replace_unks_alt(greedy_sent, list(self.other_words))

            # Construct fully random selection baseline with random within-vocab words at in-vocab positions and random out-of-vocab words at out-of-vocab positions
            random_sent = " ".join([
                random.choice(list(self.top_words_map.keys())).lower() if not missing_mask[i] else random.choice(list(self.other_words)).lower() for i in range(len(greedy_sent.split()))
            ])

            self.test_step_outputs.append({
                "true_sent": true_sent,
                "beam_sent": beam_sent,
                "beam_sent_filled": beam_sent_filled,
                "beam_sent_random_filled": beam_sent_random_filled,
                "greedy_sent": greedy_sent,
                "greedy_random_filled": greedy_random_filled,
                "random_sent": random_sent,
                "beam_masked": beam_masked,
                "prediction_info": prediction_info,
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
        parser.add_argument("--lm_weight", type=float, default=1.5)
        parser.add_argument("--har_type", type=str, default='spatial_attention')
        parser.add_argument("--pretrained_transformer", action='store_true', default=False)
        parser.add_argument("--pretrained_encoder", action='store_true', default=False)
        parser.add_argument("--embedding_dim", type=int, default=1024)
        parser.add_argument("--post_proc", action='store_true', default=False)
        parser.add_argument("--limit_context", type=int, default=8)
        parser.add_argument("--greedy_only", action='store_true', default=False)
        parser.add_argument("--no_llm_api", action='store_true', default=False)
        parser.add_argument("--random_noise_inputs", action='store_true', default=False)
        parser.add_argument("--use_kenlm", action='store_true', default=False)
        return parent_parser