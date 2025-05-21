import kenlm
import subprocess
import os
from collections import defaultdict

# Step 1: Install KenLM if not already installed
# pip install https://github.com/kpu/kenlm/archive/master.zip

# Step 2: Build n-gram model with KenLM tools
def build_kenlm_model(corpus_path, n=4, output_path="model.arpa"):
    """Build a KenLM language model from corpus"""
    # Preprocess corpus if needed (tokenization, etc.)
    # ...
    
    # Build model using KenLM command-line tools
    subprocess.run([
        "lmplz", 
        "-o", str(n),           # Order of n-gram
        "--text", corpus_path,  # Input file
        "--arpa", output_path   # Output ARPA file
    ])
    
    # Optionally convert to binary format for faster loading
    binary_path = output_path + ".binary"
    subprocess.run([
        "build_binary", 
        output_path, binary_path
    ])
    
    return binary_path  # Return path to binary model

# Step 3: Load and use the model
def get_next_word_distribution(model, context, vocab_file=None):
    """
    Get probability distribution for next word given context
    
    Args:
        model: KenLM model object
        context: List of tokens forming the context
        vocab_file: File with vocabulary (one word per line) or None to use words from context
    
    Returns:
        Dict mapping words to their probabilities
    """
    # Format context correctly
    context_str = " ".join(context)
    
    # Load vocabulary (if provided) or use words from training
    vocabulary = []
    if vocab_file and os.path.exists(vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocabulary = [line.strip() for line in f]
    else:
        # Without a vocab file, we can only score words we know about
        # This is just a simple example - in practice, you'd want a complete vocab
        vocabulary = list(set(context))  # This is just for illustration
        with open("corpus.txt", 'r', encoding='utf-8') as f:
            vocabulary = list(set(f.read().lower().split()))
        # vocabulary = ["fox", "jump", "Holmes", "Watson"]
    
    # Calculate probabilities efficiently
    distribution = {}
    for word in vocabulary:
        # Create sequence with this word appended to context
        sequence = context_str + " " + word
        
        # Get sequence probability and context probability
        sequence_log_prob = model.score(sequence, bos=False, eos=False)
        context_log_prob = model.score(context_str, bos=False, eos=False)
        
        # Convert from log space and calculate conditional probability
        # P(word|context) = P(context,word) / P(context)
        word_prob = 10 ** (sequence_log_prob - context_log_prob)
        distribution[word] = word_prob
    
    return distribution

# Usage example
def main():
    # 1. Build model (or skip if already built)
    model_path = build_kenlm_model("corpus.txt", n=4)
    
    # 2. Load model
    model = kenlm.Model(model_path)
    
    # 3. Get next word distribution
    context = ["my", "watson", "is"]
    distribution = get_next_word_distribution(model, context, "vocabulary.txt")
    
    # 4. Show top predictions
    top_words = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, prob in top_words:
        print(f"{word}: {prob:.6f}")

if __name__ == "__main__":
    main()