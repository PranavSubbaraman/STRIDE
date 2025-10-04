"""
Quick test to verify draft model is actually faster than target model.
"""
import torch
import time
import argparse

# Add project to path
import sys
sys.path.append('.')

from models.timer_xl import Model as TimerXL
from models.timer_xl_draft import Model as TimerXLDraft


class Args:
    """Mock args for model initialization"""
    input_token_len = 96
    output_token_len = 96
    dropout = 0.0
    output_attention = False
    covariate = False
    activation = 'gelu'
    use_norm = True
    flash_attention = False
    
    # Target model config
    d_model = 1024
    n_heads = 8
    d_ff = 2048
    e_layers = 1
    
    # Draft scaling
    draft_scale_d_model = 0.01
    draft_scale_n_heads = 0.01
    draft_scale_d_ff = 0.01
    draft_scale_e_layers = 0.01


def test_model_speed(model, name, batch_size=32, seq_len=672, n_channels=7, n_iters=50):
    """Time a model's forward pass"""
    model.eval()
    model = model.cuda()
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, n_channels).cuda()
    x_mark = torch.zeros(batch_size, seq_len, 1).cuda()
    y_mark = torch.zeros(batch_size, seq_len, 1).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, x_mark, y_mark)
    
    # Time
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(x, x_mark, y_mark)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    
    avg_time = sum(times) / len(times)
    print(f"\n{name}:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Avg time: {avg_time*1000:.3f} ms")
    print(f"  Per batch: {avg_time:.6f} s")
    return avg_time


def main():
    print("Testing Timer-XL vs Draft Model Speed")
    print("=" * 60)
    
    args = Args()
    
    # Build target model
    print("\nBuilding Target Model (timer_xl)...")
    target_model = TimerXL(args)
    target_time = test_model_speed(target_model, "Target Model (timer_xl)")
    
    # Build draft model
    print("\nBuilding Draft Model (timer_xl_draft with 0.01 scaling)...")
    draft_model = TimerXLDraft(args)
    draft_time = test_model_speed(draft_model, "Draft Model (timer_xl_draft)")
    
    print("\n" + "=" * 60)
    print(f"Speedup: {target_time / draft_time:.2f}x")
    print(f"Expected speedup: ~64x (based on d_model reduction)")
    
    if draft_time >= target_time * 0.9:
        print("\n⚠️  WARNING: Draft model is NOT significantly faster!")
        print("   This suggests the model may not be properly scaled.")
    else:
        print("\n✓ Draft model is faster as expected!")


if __name__ == "__main__":
    main()
