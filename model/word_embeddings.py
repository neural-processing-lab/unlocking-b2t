import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer

def generate_word_embeddings(vocab, dataset, vocab_size, layer=12):

    try:
        embeddings = torch.load(f'word_embeddings_{dataset}_{vocab_size}.pt')
        return embeddings
    except Exception:
        pass

    t5 = T5EncoderModel.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')

    """Pre-compute T5 embeddings for all words in vocabulary"""
    embeddings = []
    with torch.no_grad():
        for word in vocab:
            word = word.lower()
            tokens = tokenizer(word, return_tensors='pt', padding=True)
            outputs = t5(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer][..., :-1, :] # Ignore the last token (</s>)
            # Use mean pooling over token embeddings
            emb = hidden_states.mean(dim=1)
            embeddings.append(emb)

    del t5
    del tokenizer

    embeddings = torch.cat(embeddings, dim=0)  # [vocab_size, 1024?]

    torch.save(embeddings, f'word_embeddings_{dataset}_{vocab_size}.pt')

    return embeddings
