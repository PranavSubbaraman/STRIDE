"""
Check all draft model checkpoints to identify their actual dimensions.
"""
import torch
import os
from pathlib import Path

checkpoint_dir = Path("checkpoints")

print("=" * 90)
print("Draft Model Checkpoint Analysis")
print("=" * 90)
print()

# Find all draft checkpoints
draft_checkpoints = []
for ckpt_path in checkpoint_dir.glob("*draft*/checkpoint.pth"):
    if "ETTh1" in str(ckpt_path):
        draft_checkpoints.append(ckpt_path)

print(f"Found {len(draft_checkpoints)} ETTh1 draft checkpoints\n")

# Analyze each
results = []
for ckpt_path in sorted(draft_checkpoints):
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # Get dimensions
        d_model = state_dict['embedding.weight'].shape[0]
        n_heads_emb = state_dict['blocks.attn_layers.0.attention.inner_attention.attn_bias.emb.weight'].shape[1]
        d_ff = state_dict['blocks.attn_layers.0.conv1.weight'].shape[0]
        
        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values())
        
        # Calculate scaling
        scaling = d_model / 1024.0
        
        results.append({
            'path': ckpt_path,
            'd_model': d_model,
            'n_heads': n_heads_emb,
            'd_ff': d_ff,
            'params': total_params,
            'scaling': scaling,
        })
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")

# Sort by d_model for easy comparison
results.sort(key=lambda x: x['d_model'])

# Print results
print(f"{'Scaling':<10} {'d_model':<10} {'n_heads':<10} {'d_ff':<10} {'Params':<12} {'Directory'}")
print("-" * 90)

for r in results:
    dir_name = r['path'].parent.name[:60]
    print(f"{r['scaling']:<10.3f} {r['d_model']:<10} {r['n_heads']:<10} {r['d_ff']:<10} {r['params']:<12,} {dir_name}")

print("=" * 90)
print()
print("RECOMMENDATION:")
print()

# Find 0.5x model
model_05x = [r for r in results if 0.45 <= r['scaling'] <= 0.55]
if model_05x:
    print("✓ Found 0.5x scaled model(s):")
    for r in model_05x:
        print(f"  - {r['path']}")
        print(f"    d_model={r['d_model']}, params={r['params']:,}")
else:
    print("✗ No 0.5x scaled model found!")
    print("  You may need to train one using draft_scale_d_model=0.5")

print()

# Find what was likely used (based on error message showing d_model=64)
model_064 = [r for r in results if r['d_model'] == 64]
if model_064:
    print("⚠️  Checkpoint with d_model=64 (from error message):")
    for r in model_064:
        print(f"  - {r['path']}")
        print(f"    This is a {r['scaling']:.3f}x model - too small!")
