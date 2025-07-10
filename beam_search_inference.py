import torch, math, os, json
import torch.nn as nn
from model import *
from tokenizer import Tokenizer
from Data import Data

#choose device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f'Device set to {device}')

#load saved checkpoint
model = Transformer.create_from_checkpoint('Best_Checkpoint.pth.tar').to(device)

#get data for initial tokens
vocab_file = 'vocab_chars.json'
file = open(vocab_file, 'r')
vocab = json.loads(file.read())
file.close()
tokenizer = Tokenizer(vocab)

# Get dummy stating string
data = Data('tbbt_train.txt', 'tbbt_test.txt', tokenizer, model.ctx_window_length, sample_data=True)
initial_tokens = data._encoded_train_data[-data.ctx_size:]
initial_text = data.train_text[0:data.ctx_size]



# Generation Config
TEMPERATURE = 0.8
TOP_P = 1
BEAM_WIDTH = 32
GEN_LENGTH = 2000
NGRAM_PENALTY = 3

model.eval()

# Beam: list of (tokens, text, total_log_prob, seen_ngrams)
beam = [(initial_tokens, initial_text, 0.0, set())]

# Top-p sampling filter
def top_p_filter(probs, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff = cumulative_probs > p
    if torch.any(cutoff):
        last_index = torch.where(cutoff)[0][0]
        return sorted_probs[:last_index + 1], sorted_indices[:last_index + 1]
    return sorted_probs, sorted_indices

for step in range(GEN_LENGTH):
    all_candidates = []
    x_batch = []
    beam_metadata = []

    for tokens, text, log_prob_sum, seen_ngrams in beam:
        x = torch.tensor(tokens[-data.ctx_size:]).to(device).reshape(1, -1)
        x_batch.append(x)
        beam_metadata.append((tokens, text, log_prob_sum, seen_ngrams))

    x_batch = torch.cat(x_batch, dim=0)

    with torch.no_grad():
        output = model(x_batch)
        output = output.view((x_batch.size(0), -1, len(tokenizer.vocab_list)))
        logits = output[:, -1, :]
        probs = nn.functional.softmax(logits / TEMPERATURE, dim=-1)

    for b_idx in range(len(beam)):
        tokens, text, log_prob_sum, seen_ngrams = beam_metadata[b_idx]
        filtered_probs, filtered_indices = top_p_filter(probs[b_idx], TOP_P)

        for i in range(len(filtered_indices)):
            token_id = filtered_indices[i].item()
            prob = filtered_probs[i].item()
            if prob <= 0: continue

            new_tokens = tokens + [token_id]
            new_char = tokenizer.decode([token_id])[0]
            new_text = text + new_char
            new_log_prob_sum = log_prob_sum + math.log(prob)

            # n-gram repetition penalty
            ngram = tuple(new_tokens[-NGRAM_PENALTY:])
            repeated = ngram in seen_ngrams
            penalty = -1.0 if repeated else 0.0
            new_seen_ngrams = seen_ngrams.copy()
            new_seen_ngrams.add(ngram)

            all_candidates.append((
                new_tokens,
                new_text,
                new_log_prob_sum + penalty,
                new_seen_ngrams
            ))

    # Keep best beam paths
    all_candidates.sort(key=lambda x: x[2] / len(x[0]), reverse=True)
    beam = all_candidates[:BEAM_WIDTH]

    # Best candidate
    best_tokens, best_text, best_log_prob, _ = beam[0]
    avg_log_prob = best_log_prob / len(best_tokens)
    perplexity = math.exp(-avg_log_prob)

    # Clear terminal and show progress
    os.system('cls' if os.name == 'nt' else 'clear')
    print('"' + best_text + '"\n', flush=True)
    print(f"[Step {step+1}/{GEN_LENGTH}] Perplexity: {perplexity:.3f}", flush=True)

# Final output
print("\n=== Final Output ===")
print(best_text)
print(f"[Final Perplexity: {perplexity:.3f}]")