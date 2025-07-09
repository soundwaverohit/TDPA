#!/usr/bin/env python3
import argparse, random, math, numpy as np, pandas as pd, torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from adapter_gpt import VQEncoder, VQDecoder, Denoiser

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_models(device, adapter_path):
    # load tokenizer & frozen GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    # load adapter weights
    state = torch.load(adapter_path, map_location=device)
    vocab_size = model.config.vocab_size

    encoder  = VQEncoder(vocab_size=vocab_size).to(device)
    decoder  = VQDecoder(vocab_size=vocab_size).to(device)
    denoiser = Denoiser().to(device)

    encoder.load_state_dict(state["encoder"]);   encoder.eval()
    decoder.load_state_dict(state["decoder"]);   decoder.eval()
    denoiser.load_state_dict(state["denoiser"]); denoiser.eval()

    return tokenizer, model, encoder, decoder, denoiser

def base_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits[0, -1]

def tpda_refine(logits, encoder, decoder, denoiser, alpha, p_list):
    c = encoder(logits.unsqueeze(0)).squeeze(0)
    noisy = c.clone()
    for p in p_list:
        mask = torch.bernoulli(p * torch.ones_like(noisy))
        noisy = (noisy + mask) % 2
    c_hat     = (denoiser(noisy.unsqueeze(0)).squeeze(0) > 0.5).float()
    logits_hat = decoder(c_hat.unsqueeze(0)).squeeze(0)
    return (1 - alpha) * logits + alpha * logits_hat

def generate(tokenizer, model, encoder, decoder, denoiser,
             device, prompt, alpha, p_list, max_len=30, use_tpda=False):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_len):
        logits = base_logits(model, input_ids)
        if use_tpda:
            logits = tpda_refine(logits, encoder, decoder, denoiser, alpha, p_list)
        next_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def qualitative(tokenizer, model, encoder, decoder, denoiser,
                device, prompts, alpha, p_list):
    print("\n=== Qualitative Comparison ===")
    for prompt in prompts:
        print(f"\n> Prompt: {prompt}")
        base_out = generate(tokenizer, model, encoder, decoder, denoiser,
                            device, prompt, alpha, p_list, use_tpda=False)
        tpda_out = generate(tokenizer, model, encoder, decoder, denoiser,
                            device, prompt, alpha, p_list, use_tpda=True)
        print("Base GPT-2 : ", base_out)
        print("TPDA-Refined:", tpda_out)

def quantitative(tokenizer, model, encoder, decoder, denoiser,
                 device, df, eval_frac, alpha, p_list):
    print("\n=== Quantitative Evaluation ===")
    df_eval = df.sample(frac=eval_frac, random_state=42)
    heldout = list(zip(df_eval["question"], df_eval["answer"]))

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

            corr_b += int(torch.argmax(probs_b).item() == true_id)
            corr_t += int(torch.argmax(probs_t).item() == true_id)
            top5_b += int(true_id in torch.topk(probs_b, 5).indices)
            top5_t += int(true_id in torch.topk(probs_t, 5).indices)

            context.append(true_id)

    ppl_b = math.exp(-sum_log_b / total_tok)
    ppl_t = math.exp(-sum_log_t / total_tok)
    acc_b = corr_b / total_tok
    acc_t = corr_t / total_tok
    t5_b  = top5_b / total_tok
    t5_t  = top5_t / total_tok

    print(f"Tokens eval’d:        {total_tok}")
    print(f"Perplexity (↓ better): Base = {ppl_b:.2f}, TPDA = {ppl_t:.2f}")
    print(f"Next-Token Acc.       : Base = {acc_b:.2%}, TPDA = {acc_t:.2%}")
    print(f"Top-5 Acc.            : Base = {t5_b:.2%}, TPDA = {t5_t:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 ± TPDA")
    parser.add_argument("--adapter", type=str, default="adapter_gpt2.pt",
                        help="Path to trained adapter checkpoint")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Blend factor α")
    parser.add_argument("--p_list", type=float, nargs="+",
                        default=[0.01, 0.02, 0.05],
                        help="List of bit-flip probs per micro-step")
    parser.add_argument("--eval_frac", type=float, default=0.3,
                        help="Fraction of data to hold out")
    parser.add_argument("--qual", nargs="+",
                        default=["What is hypertension"],
                        help="Prompts for qualitative tests")
    args = parser.parse_args()

    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, encoder, decoder, denoiser = load_models(device, args.adapter)

    qualitative(tokenizer, model, encoder, decoder, denoiser,
                device, args.qual, args.alpha, args.p_list)

    df = pd.read_csv("data.csv")
    quantitative(tokenizer, model, encoder, decoder, denoiser,
                 device, df, args.eval_frac, args.alpha, args.p_list)

if __name__ == "__main__":
    main()
