#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from adapter_gpt import VQEncoder, VQDecoder, Denoiser

class GPT2LogitDataset(Dataset):
    def __init__(self, data, tokenizer, model, device):
        self.examples = []
        for q, a in data:
            prompt_ids = tokenizer.encode(q, add_special_tokens=False)
            target_ids = tokenizer.encode(a, add_special_tokens=False)
            for i, true_id in enumerate(target_ids):
                input_ids = torch.tensor([prompt_ids + target_ids[:i]]).to(device)
                with torch.no_grad():
                    logits = model(input_ids).logits[0, -1]
                self.examples.append((logits.cpu(), true_id))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def train_adapter(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load base GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model     = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    model.eval()

    # 2) Prepare data
    df = pd.read_csv(args.data_csv) #pd.read_csv("hf://datasets/keivalya/MedQuad-MedicalQnADataset/medDataset_processed.csv")
    #df = df[:1000]
    #pd.read_csv(args.data_csv)
    data = list(zip(df["question"], df["answer"]))
    ds   = GPT2LogitDataset(data, tokenizer, model, device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # 3) Train VQ encoder + decoder
    encoder = VQEncoder(vocab_size=model.config.vocab_size,
                        hidden=args.hidden_dim,
                        code_dim=args.code_dim).to(device)
    decoder = VQDecoder(vocab_size=model.config.vocab_size,
                        hidden=args.hidden_dim,
                        code_dim=args.code_dim).to(device)
    opt_vq  = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr
    )
    mse = nn.MSELoss()
    print("Training VQ...")
    for epoch in range(args.vq_epochs):
        total_loss = 0.0
        for logits, _ in loader:
            logits = logits.to(device)
            opt_vq.zero_grad()
            code  = encoder(logits)
            recon = decoder(code)
            loss  = mse(recon, logits)
            loss.backward()
            opt_vq.step()
            total_loss += loss.item()
        print(f" VQ Epoch {epoch+1}/{args.vq_epochs}: {total_loss/len(loader):.4f}")

    # 4) Build clean code dataset
    all_codes = []
    with torch.no_grad():
        for logits, _ in loader:
            all_codes.append(encoder(logits.to(device)))
    clean = torch.cat(all_codes, dim=0).cpu()

    # 5) Create noisy codes via T-step diffusion
    noisy = clean.clone()
    for _ in range(args.diffusion_steps):
        mask  = torch.bernoulli(args.noise_p * torch.ones_like(noisy))
        noisy = (noisy + mask) % 2

    # 6) Train denoiser
    denoiser = Denoiser(code_dim=args.code_dim,
                        hidden=args.hidden_dim).to(device)
    opt_den  = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    bce      = nn.BCELoss()
    den_ds   = torch.utils.data.TensorDataset(noisy, clean)
    den_loader = DataLoader(den_ds, batch_size=args.batch_size, shuffle=True)
    print("Training Denoiser...")
    for epoch in range(args.den_epochs):
        total_loss = 0.0
        for n, c in den_loader:
            n, c = n.to(device), c.to(device)
            opt_den.zero_grad()
            pred = denoiser(n)
            loss = bce(pred, c)
            loss.backward()
            opt_den.step()
            total_loss += loss.item()
        print(f" Denoiser Epoch {epoch+1}/{args.den_epochs}: {total_loss/len(den_loader):.4f}")

    # 7) Save adapter weights
    torch.save({
        "encoder":  encoder.cpu().state_dict(),
        "decoder":  decoder.cpu().state_dict(),
        "denoiser": denoiser.cpu().state_dict()
    }, args.output_path)
    print(f"Adapter saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TPDA adapter")
    parser.add_argument("--data_csv",    type=str,   default="data.csv",
                        help="CSV with columns 'question','answer'")
    parser.add_argument("--model_name",  type=str,   default="gpt2")
    parser.add_argument("--code_dim",    type=int,   default=512)
    parser.add_argument("--hidden_dim",  type=int,   default=512)
    parser.add_argument("--vq_epochs",   type=int,   default=10)
    parser.add_argument("--den_epochs",  type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--noise_p",     type=float, default=0.1,
                        help="Bit-flip probability")
    parser.add_argument("--diffusion_steps", type=int, default=5,
                        help="Number of micro-steps")
    parser.add_argument("--output_path", type=str,   default="adapter_gpt2.pt")
    args = parser.parse_args()
    train_adapter(args)
