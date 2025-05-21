import math
import os
import torch
import threading
import queue
import math
import kenlm

from string import punctuation

from transformers import AutoModelForCausalLM, AutoTokenizer

kenlm_model_path = "model.arpa.binary"
if os.path.exists(kenlm_model_path):
    kenlm_model = kenlm.Model(kenlm_model_path)
else:
    kenlm_model = None

# Use a smaller Llama variant for speed if possible
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda")
model.eval()

def get_llama_lm_score(sequence):

    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt").to("cuda")
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get the last token's prediction (what comes next)
    last_token_logits = logits[0, -1, :]
    
    # Convert to probabilities
    probs = torch.log_softmax(last_token_logits, dim=0)

    return probs

def process_beam_batch(beam_batch, position, asr_predictions, vocabulary, beam_width, lm_weight, result_queue, is_missing=False, predict_missing=False, limit_context=None, use_kenlm=False):
    candidates = []
    scaling = math.log(len(vocabulary) / model.config.vocab_size)
    for beam in beam_batch:
        prev_sequence = beam["sequence"]
        prev_score = beam["score"]

        if is_missing:

            if predict_missing:

                with torch.no_grad():
                    if limit_context is not None:
                        probs = get_llama_lm_score(" ".join(prev_sequence.split()[-limit_context:]))
                    else:
                        probs = get_llama_lm_score(prev_sequence)

                # Get the top beam_width candidates from these probs
                top_probs, top_indices = torch.sort(probs, descending=True)

                # Remove candidates that are not words in our expected vocabulary
                for index, prob in zip(top_indices, top_probs):
                    candidate = tokenizer.decode(index).strip().lower()
                    if not any(p in candidate for p in punctuation):
                        candidates.append({
                            "sequence": prev_sequence + (" " if prev_sequence else "") + candidate,
                            "score": prev_score + lm_weight * (prob + scaling)
                        })
                    if len(candidates) >= beam_width:
                        break
                
                if len(candidates) < beam_width:
                    print("Warning: failed to find candidate for missing word... skipping this position.")
                    while len(candidates) < beam_width:
                        candidates.append({
                            "sequence": prev_sequence,
                            "score": prev_score
                        })
            else:
                # If we are not predicting missing words, just skip this position
                candidates.append({
                    "sequence": prev_sequence + (" " if prev_sequence else "") + "[UNK]",
                    "score": prev_score
                })

        else:

            if not use_kenlm:
                with torch.no_grad():
                    if limit_context is not None:
                        probs = get_llama_lm_score(" ".join(prev_sequence.split()[-limit_context:]))
                    else:
                        probs = get_llama_lm_score(prev_sequence)
            
            # Get top N words from ASR for this position
            top_indices = torch.argsort(asr_predictions[position])
            top_words = [(vocabulary[idx], asr_predictions[position][idx]) for idx in top_indices[-beam_width:]]
            
            for word, asr_log_prob in top_words:

                if use_kenlm:
                    sequence = prev_sequence + " " + word
                    # Get sequence probability and context probability
                    sequence_log_prob = kenlm_model.score(sequence, bos=False, eos=False)
                    context_log_prob = kenlm_model.score(prev_sequence, bos=False, eos=False)

                    # Convert from log space and calculate conditional probability
                    # P(word|context) = P(context,word) / P(context)
                    lm_log_prob = sequence_log_prob - context_log_prob
                else:
                    next_word_tokens = tokenizer.encode(" " + word)[1:]  # Skip the BOS token
                    lm_log_prob = probs[next_word_tokens[0]].item()
                    
                    # Scale LM score based on vocabulary size
                    lm_log_prob += scaling
                
                # Combine scores (log domain)
                new_score = prev_score + asr_log_prob + lm_weight * lm_log_prob
                
                candidates.append({
                    "sequence": prev_sequence + (" " if prev_sequence else "") + word,
                    "score": new_score
                })
    
    result_queue.put(candidates)

def beam_search_with_llama_threaded(asr_predictions, vocabulary, missing_mask=None, predict_missing=False, beam_width=10, lm_weight=0.5, num_threads=4, limit_context=None, use_kenlm=False):
    beams = [{"sequence": "", "score": 0.0}]
    asr_predictions = torch.log_softmax(asr_predictions, dim=-1)
    
    for position in range(len(asr_predictions)):
        # Split beams into batches for parallel processing
        batch_size = max(1, len(beams) // num_threads)
        beam_batches = [beams[i:i + batch_size] for i in range(0, len(beams), batch_size)]
        
        # Create a queue to collect results
        result_queue = queue.Queue()

        if missing_mask is not None and missing_mask[position]:
            is_missing = True
        else:
            is_missing = False
        
        # Process beam batches in parallel using threads
        threads = []
        for beam_batch in beam_batches:
            thread = threading.Thread(
                target=process_beam_batch,
                args=(
                    beam_batch, position, asr_predictions, vocabulary,
                    beam_width, lm_weight, result_queue, is_missing, predict_missing,
                    limit_context, use_kenlm
                )
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect all candidates
        all_candidates = []
        while not result_queue.empty():
            all_candidates.extend(result_queue.get())
        
        # Keep top beam_width candidates
        beams = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:beam_width]
    
    return beams[0]["sequence"]

if __name__ == "__main__":
    # Create a simple test vocabulary
    test_vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "jumped", "over"]
    
    # Mock ASR predictions (batch of log probabilities for each position)
    # For each position, we have a tensor of log probabilities for each word in vocab
    mock_asr_predictions = torch.stack([
        # Position 1: "the" has highest probability
        torch.tensor([-1.0, -5.0, -6.0, -4.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]),
        # Position 2: "cat" and "dog" are both plausible
        torch.tensor([-8.0, -1.5, -9.0, -7.0, -9.0, -1.4, -9.0, -10.0, -11.0, -12.0]),
        # Position 3: "sat" and "ran" are plausible
        torch.tensor([-10.0, -10.0, -1.2, -7.0, -9.0, -10.0, -1.0, -10.0, -1.8, -12.0]),
        # Position 4: "on" and "fast" is most likely
        torch.tensor([-10.0, -10.0, -10.0, -1.0, -9.0, -10.0, -10.0, -0.5, -10.0, -12.0]),
        # Position 5: "mat" has highest probability
        torch.tensor([-10.0, -10.0, -10.0, -8.0, -0.8, -10.0, -10.0, -10.0, -10.0, -10.0]),
    ])
    
    # Test with different beam widths
    print("Testing beam search with different beam widths:")
    for beam_width in [1, 3, 5, 10]:
        result = beam_search_with_llama_threaded(
            mock_asr_predictions, 
            test_vocab,
            beam_width=beam_width,
            lm_weight=0.5,
            missing_mask=[False, False, True, False, False],
            predict_missing=False,
            use_kenlm=True,
        )
        print(f"Beam width {beam_width}: {result}")
    
    # Test with different language model weights
    print("\nTesting beam search with different LM weights:")
    for lm_weight in [0.0, 0.5, 1.0, 2.0]:
        result = beam_search_with_llama_threaded(
            mock_asr_predictions, 
            test_vocab,
            beam_width=10,
            lm_weight=lm_weight,
            use_kenlm=True,
        )
        print(f"LM weight {lm_weight}: {result}")

    # Test with sequence of length 64
    print("\nTesting beam search with sequence length 64:")
    mock_asr_predictions_64 = torch.randn(64, 100)
    result = beam_search_with_llama_threaded(
        mock_asr_predictions_64, 
        test_vocab * 10,  # Mock vocabulary of size 100
        beam_width=5,
        lm_weight=0.5
    )
    print("Result for sequence length 64:", result)
    
    print("\nNote: In a real implementation, you would need more sophisticated")
    print("tokenization and handling of subwords for accurate testing.")