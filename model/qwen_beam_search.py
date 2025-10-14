import torch
import numpy as np
from typing import List, Optional
from vllm import LLM, SamplingParams

# Global variable to hold the LLM instance
_llm: Optional[LLM] = None

# Prefix to prepend to all prompts
PROMPT_PREFIX = "/no_think "


def get_llm() -> LLM:
    """Lazy-load the vLLM model to avoid initialization issues with multiprocessing."""
    global _llm
    if _llm is None:
        _llm = LLM(
            model="Qwen/Qwen3-0.6B",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            max_model_len=512,
        )
    return _llm


def get_sequence_log_probs(sequences: List[str]) -> np.ndarray:
    """
    Compute log probabilities for a batch of sequences using vLLM.

    Args:
        sequences: List of text sequences to score

    Returns:
        Array of log probabilities, one per sequence
    """
    # Prepend the no_think prefix to all sequences
    full_sequences = [PROMPT_PREFIX + seq for seq in sequences]

    # Create sampling params that request prompt logprobs
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,  # We just need the prompt logprobs
        prompt_logprobs=1,  # Request logprobs for all prompt tokens
    )

    # Get outputs from vLLM
    llm = get_llm()
    outputs = llm.generate(full_sequences, sampling_params)

    log_probs = []
    for output in outputs:
        # Sum up all the prompt token log probabilities
        # prompt_logprobs is a list of dicts, one per token
        if output.prompt_logprobs:
            # Skip the first token (usually BOS) and sum the rest
            token_logprobs = [
                list(token_dict.values())[0].logprob
                for token_dict in output.prompt_logprobs[1:]
                if token_dict
            ]
            total_logprob = sum(token_logprobs) if token_logprobs else 0.0
        else:
            total_logprob = 0.0
        log_probs.append(total_logprob)

    return np.array(log_probs)


def normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
    """
    Normalize log probabilities to [0, 1] using softmax.

    Args:
        log_probs: Array of log probabilities

    Returns:
        Normalized probabilities in [0, 1]
    """
    # Convert log probs to probs using log-sum-exp trick for numerical stability
    max_log_prob = np.max(log_probs)
    exp_probs = np.exp(log_probs - max_log_prob)
    normalized_probs = exp_probs / np.sum(exp_probs)
    return normalized_probs


def qwen_beam_search(
    asr_predictions: torch.Tensor,
    vocabulary: List[str],
    beam_width: int = 10,
    num_return_sequences: int = 1,
    temperature: float = 1.0,
) -> List[str]:
    """
    Stochastic beam search with Qwen language model using sequence-level probabilities.

    Args:
        asr_predictions: ASR predictions [seq_len, vocab_size] (raw logits or log probs)
        vocabulary: List of vocabulary words
        beam_width: Number of beams to maintain and candidates to evaluate per position
        num_return_sequences: Number of top sequences to return (default: 1 for backward compatibility)
        temperature: Temperature for sampling diversity (>1 = more diverse, <1 = more focused)

    Returns:
        List of top num_return_sequences predicted sequences as strings
    """
    # Initialize beams
    beams = [{"sequence": "", "log_prob": 0.0}]

    # Apply temperature scaling to logits BEFORE softmax
    if temperature == 0.0:
        # No scaling for deterministic case
        scaled_logits = asr_predictions
    else:
        scaled_logits = asr_predictions / temperature

    # Ensure ASR predictions are log probabilities
    asr_log_probs = torch.log_softmax(scaled_logits, dim=-1)

    # Also get regular probabilities for sampling (only needed for stochastic case)
    if temperature > 0.0:
        asr_probs = torch.softmax(scaled_logits, dim=-1)
    else:
        asr_probs = None

    for position in range(len(asr_predictions)):
        # Collect all candidate sequences from all beams
        all_sequences = []
        all_asr_log_probs = []

        top_k = min(beam_width, len(vocabulary))

        # Deterministic case: all beams share the same top-K candidates
        if temperature == 0.0:
            topk_indices = torch.argsort(asr_log_probs[position], descending=True)[:top_k]

            for beam in beams:
                prev_sequence = beam["sequence"]
                prev_log_prob = beam["log_prob"]

                for idx in topk_indices:
                    word = vocabulary[idx.item()]
                    asr_log_prob = asr_log_probs[position, idx].item()

                    # Construct new sequence
                    new_sequence = prev_sequence + (" " if prev_sequence else "") + word
                    all_sequences.append(new_sequence)
                    all_asr_log_probs.append(prev_log_prob + asr_log_prob)

        # Stochastic case: each beam samples its own candidates
        else:
            for beam in beams:
                prev_sequence = beam["sequence"]
                prev_log_prob = beam["log_prob"]

                # Each beam samples its own candidates from the probability distribution
                topk_indices = torch.multinomial(asr_probs[position], num_samples=top_k, replacement=False)

                for idx in topk_indices:
                    word = vocabulary[idx.item()]
                    asr_log_prob = asr_log_probs[position, idx].item()

                    # Construct new sequence
                    new_sequence = prev_sequence + (" " if prev_sequence else "") + word
                    all_sequences.append(new_sequence)
                    all_asr_log_probs.append(prev_log_prob + asr_log_prob)

        # Batch compute LM sequence probabilities for ALL candidates
        lm_log_probs = get_sequence_log_probs(all_sequences)

        # Also compute LM log probs for the prefix sequences (before adding new word)
        # to get incremental probabilities
        prefix_sequences = []
        for beam in beams:
            for _ in range(top_k):
                prefix_sequences.append(beam["sequence"])

        # Get LM scores for prefixes (unless this is the first position)
        if position == 0:
            # First position has no prefix
            lm_prefix_log_probs = np.zeros(len(prefix_sequences))
        else:
            lm_prefix_log_probs = get_sequence_log_probs(prefix_sequences)

        # Combine ASR and LM probabilities (both in log space)
        # Use incremental LM probability: p(w1...wn) - p(w1...wn-1)
        candidates = []
        for i, (seq, asr_log_prob, lm_log_prob) in enumerate(zip(all_sequences, all_asr_log_probs, lm_log_probs)):
            lm_incremental = lm_log_prob - lm_prefix_log_probs[i]
            combined_log_prob = asr_log_prob + lm_incremental
            candidates.append({
                "sequence": seq,
                "log_prob": combined_log_prob
            })

        # Keep top beam_width candidates
        beams = sorted(candidates, key=lambda x: x["log_prob"], reverse=True)[:beam_width]

    # Return the top num_return_sequences
    return [beam["sequence"] for beam in beams[:num_return_sequences]]


if __name__ == "__main__":
    # Simple test
    test_vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "jumped", "over"]

    # Mock ASR predictions (logits for each position)
    mock_asr_predictions = torch.stack([
        torch.tensor([-1.0, -5.0, -6.0, -4.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]),
        torch.tensor([-8.0, -1.5, -9.0, -7.0, -9.0, -1.4, -9.0, -10.0, -11.0, -12.0]),
        torch.tensor([-10.0, -10.0, -1.2, -7.0, -9.0, -10.0, -1.0, -10.0, -1.8, -12.0]),
        torch.tensor([-10.0, -10.0, -10.0, -1.0, -9.0, -10.0, -10.0, -0.5, -10.0, -12.0]),
        torch.tensor([-10.0, -10.0, -10.0, -8.0, -0.8, -10.0, -10.0, -10.0, -10.0, -10.0]),
    ])

    print("Testing Qwen beam search with vLLM:")
    result = qwen_beam_search(
        mock_asr_predictions,
        test_vocab,
        beam_width=5,
    )
    print(f"Result: {result}")
