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
    
    # Print sample keys for debugging
    print(f"\nTotal keys in checkpoint: {len(model_state)}")
    print("Sample keys (first 10):")
    for k in list(model_state.keys())[:10]:
        print(f"  {k}")

    # Extract DiT weights (net.*)
    dit_state = {k.replace("net.", "", 1): v for k, v in model_state.items() if k.startswith("net.")}
    dit_path = os.path.join(args.output_dir, "cosmos_dit.pt")
    torch.save(dit_state, dit_path)
    print(f"\nSaved DIT weights ({len(dit_state)} keys) to {dit_path}")

    # Extract VAE weights - try multiple possible prefixes
    vae_keys = []
    vae_prefixes = ["tokenizer.", "vae.", "encoder.", "model.tokenizer.", "model.vae."]
    
    for prefix in vae_prefixes:
        matching_keys = [k for k in model_state.keys() if k.startswith(prefix)]
        if matching_keys:
            vae_keys = matching_keys
            print(f"\nFound VAE weights with prefix '{prefix}': {len(vae_keys)} keys")
            # Remove prefix and save
            if prefix == "encoder." or prefix == "decoder.":
                # Keep the prefix as is for these cases
                vae_state = {k: v for k, v in model_state.items() if k.startswith(prefix)}
            else:
                vae_state = {k.replace(prefix, "", 1): v for k, v in model_state.items() if k.startswith(prefix)}
            break
    
    if not vae_keys:
        # Try to find any keys that might be VAE related
        print("\nCould not find VAE weights with known prefixes. Looking for 'encoder' or 'decoder' keys...")
        vae_state = {}
        for k, v in model_state.items():
            if any(x in k for x in ["encoder", "decoder", "conv1", "conv2"]):
                if not k.startswith("net."):  # Exclude DiT
                    vae_state[k] = v
        print(f"Found {len(vae_state)} potential VAE keys")
    
    if vae_state:
        vae_path = os.path.join(args.output_dir, "cosmos_vae.pt")
        torch.save(vae_state, vae_path)
        print(f"Saved VAE weights ({len(vae_state)} keys) to {vae_path}")
        print("Sample VAE keys (first 10):")
        for k in list(vae_state.keys())[:10]:
            print(f"  {k}")
    else:
        print("WARNING: No VAE weights found!")

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
