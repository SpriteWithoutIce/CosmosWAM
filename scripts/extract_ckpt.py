import argparse
import torch
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to consolidated cosmos .pt checkpoint")
    parser.add_argument("--output_dir", default="./checkpoints", help="Output directory for extracted weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading consolidated checkpoint from {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")

    model_state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    dit_state = {k.replace("net.", "", 1): v for k, v in model_state.items() if k.startswith("net.")}
    dit_path = os.path.join(args.output_dir, "cosmos_dit.pt")
    torch.save(dit_state, dit_path)
    print(f"Saved DIT weights ({len(dit_state)} keys) to {dit_path}")

    vae_state = {k.replace("tokenizer.", "", 1): v for k, v in model_state.items() if k.startswith("tokenizer.")}
    vae_path = os.path.join(args.output_dir, "cosmos_vae.pt")
    torch.save(vae_state, vae_path)
    print(f"Saved VAE weights ({len(vae_state)} keys) to {vae_path}")

    print("Extraction complete.")


if __name__ == "__main__":
    main()
