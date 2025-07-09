# hyperparam_search.py
"""
Parallel hyperparameter sweep for TPDA vs base GPT-2.
Records perplexity, next-token accuracy, top-5 accuracy for both models.
Uses Ray for parallel evaluation and outputs a CSV of results, plus reports best settings.
"""
import math
import random
import argparse
import pandas as pd
import numpy as np
import torch
import ray
from itertools import product
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from adapter_gpt import VQEncoder, VQDecoder, Denoiser

# --------------- Utility Functions ----------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_models(adapter_path: str, device: torch.device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    state = torch.load(adapter_path, map_location=device)

    encoder = VQEncoder(vocab_size=model.config.vocab_size).to(device)
    decoder = VQDecoder(vocab_size=model.config.vocab_size).to(device)
    denoiser = Denoiser().to(device)

    encoder.load_state_dict(state['encoder']); encoder.eval()
    decoder.load_state_dict(state['decoder']); decoder.eval()
    denoiser.load_state_dict(state['denoiser']); denoiser.eval()

    return tokenizer, model, encoder, decoder, denoiser


def base_logits(model, input_ids: torch.LongTensor):
    with torch.no_grad():
        return model(input_ids).logits[0, -1]


def tpda_refine(logits: torch.Tensor,
                encoder: VQEncoder,
                decoder: VQDecoder,
                denoiser: Denoiser,
                alpha: float,
                p_list: list):
    c = encoder(logits.unsqueeze(0)).squeeze(0)
    noisy = c.clone()
    for p in p_list:
        mask = torch.bernoulli(p * torch.ones_like(noisy))
        noisy = (noisy + mask) % 2
    c_hat = (denoiser(noisy.unsqueeze(0)).squeeze(0) > 0.5).float()
    logits_hat = decoder(c_hat.unsqueeze(0)).squeeze(0)
    return (1 - alpha) * logits + alpha * logits_hat


def evaluate_quantitative(tokenizer, model, encoder, decoder, denoiser,
                           device, df: pd.DataFrame,
                           eval_frac: float,
                           alpha: float,
                           p_list: list):
    df_eval = df.sample(frac=eval_frac, random_state=42)
    heldout = list(zip(df_eval['question'], df_eval['answer']))

    total_tok = sum_log_b = sum_log_t = 0
    corr_b = corr_t = top5_b = top5_t = 0

    for q, a in heldout:
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        a_ids = tokenizer.encode(a, add_special_tokens=False)
        context = q_ids.copy()
        for true_id in a_ids:
            input_ids = torch.tensor([context]).to(device)
            logits_b = base_logits(model, input_ids)
            logits_t = tpda_refine(logits_b, encoder, decoder, denoiser, alpha, p_list)

            probs_b = torch.softmax(logits_b, dim=-1)
            probs_t = torch.softmax(logits_t, dim=-1)

            sum_log_b += math.log(probs_b[true_id].item() + 1e-12)
            sum_log_t += math.log(probs_t[true_id].item() + 1e-12)
            total_tok += 1

            pred_b = torch.argmax(probs_b).item()
            pred_t = torch.argmax(probs_t).item()
            corr_b += int(pred_b == true_id)
            corr_t += int(pred_t == true_id)

            top5_b += int(true_id in torch.topk(probs_b, 5).indices)
            top5_t += int(true_id in torch.topk(probs_t, 5).indices)

            context.append(true_id)

    ppl_b = math.exp(-sum_log_b / total_tok)
    ppl_t = math.exp(-sum_log_t / total_tok)
    acc_b = corr_b / total_tok
    acc_t = corr_t / total_tok
    t5_b  = top5_b / total_tok
    t5_t  = top5_t / total_tok

    return {
        'ppl_base': ppl_b,
        'ppl_tpda': ppl_t,
        'acc_base': acc_b,
        'acc_tpda': acc_t,
        'top5_base': t5_b,
        'top5_tpda': t5_t
    }

# --------------- Ray Remote Task ----------------
@ray.remote
def run_trial(alpha: float, p_list: list, eval_frac: float, data_path: str, adapter_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(42)
    tokenizer, model, encoder, decoder, denoiser = load_models(adapter_path, device)
    df = pd.read_csv(data_path)
    metrics = evaluate_quantitative(
        tokenizer, model, encoder, decoder, denoiser,
        device, df, eval_frac, alpha, p_list
    )
    return {'alpha': alpha,
            'p_list': p_list,
            'eval_frac': eval_frac,
            **metrics}

# ------------------- Main Script ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for TPDA using Ray"
    )
    parser.add_argument('--data_csv', type=str, default='data.csv')
    parser.add_argument('--adapter', type=str, default='adapter_gpt2.pt')
    parser.add_argument('--output_csv', type=str, default='results.csv')
    args = parser.parse_args()

    # Define search grid
    alphas    = [0.05, 0.1, 0.2]
    p_lists   = [[0.01, 0.02, 0.05], [0.05, 0.1, 0.2]]
    eval_fracs= [0.1, 0.2, 0.3]

    ray.init(ignore_reinit_error=True)

    # Launch parallel trials
    futures = []
    for alpha, p_list, ef in product(alphas, p_lists, eval_fracs):
        futures.append(
            run_trial.remote(alpha, p_list, ef, args.data_csv, args.adapter)
        )

    results = ray.get(futures)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

    # Find best hyperparams
    best_ppl   = df.loc[df['ppl_tpda'].idxmin()]
    best_acc   = df.loc[df['acc_tpda'].idxmax()]
    best_top5  = df.loc[df['top5_tpda'].idxmax()]

    print("\nBest TPDA settings:")
    print(f"- Lowest TPDA Perplexity:    α={best_ppl['alpha']}, p_list={best_ppl['p_list']}, eval_frac={best_ppl['eval_frac']} (ppl={best_ppl['ppl_tpda']:.2f})")
    print(f"- Highest TPDA Acc (Top-1):  α={best_acc['alpha']}, p_list={best_acc['p_list']}, eval_frac={best_acc['eval_frac']} (acc={best_acc['acc_tpda']:.2%})")
    print(f"- Highest TPDA Acc (Top-5):  α={best_top5['alpha']}, p_list={best_top5['p_list']}, eval_frac={best_top5['eval_frac']} (top5={best_top5['top5_tpda']:.2%})")

if __name__ == "__main__":
    main()
