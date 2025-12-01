#!/usr/bin/env python3
"""
Generate Y_test_tensor_bf16.bin as a 64 x 768 matrix.
- Rows 0..63: start as an exact bitwise copy of X_test_tensor_bf16.bin
- Rows 46..53: then overwrite with custom definitions (same as before)

Thus, rows other than 46..53 remain identical to X; rows 46..53 follow Y's patterns.
"""

import os
import sys
import numpy as np
import torch

# Constants
ROWS = 64
COLS = 768
SEED = 12345  # deterministic base seed; we use SEED+original_row_index

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
X_FILE = os.path.join(DATA_DIR, 'X_test_tensor_bf16.bin')
Y_FILE = os.path.join(DATA_DIR, 'Y_test_tensor_bf16.bin')

# bf16 bit-pattern helpers (vectorized)
def torch_bf16_to_uint16(t: torch.Tensor) -> np.ndarray:
    f32 = t.float().cpu().numpy().astype(np.float32, copy=False)
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)

def bf16_from_bits_u16(bits_u16: np.ndarray) -> torch.Tensor:
    bits_u16 = np.asarray(bits_u16, dtype=np.uint16)
    bits_u32 = bits_u16.astype(np.uint32) << 16
    f32 = bits_u32.view(np.float32)
    return torch.from_numpy(f32).to(torch.bfloat16)

# Common bf16 constants
BF16_POS_ZERO   = 0x0000
BF16_INF_POS    = 0x7F80
BF16_INF_NEG    = 0xFF80
BF16_QNAN       = 0x7FC0
BF16_MAX        = 0x7F7F
BF16_MIN_NORMAL = 0x0080


def main():
    total = ROWS * COLS

    # Read X and initialize Y as a full copy
    if not os.path.exists(X_FILE):
        print(f"ERROR: Missing X file: {X_FILE}\nPlease run generate_X_test_tensor_bf16.py first.")
        sys.exit(1)
    x_u16 = np.fromfile(X_FILE, dtype=np.uint16)
    if x_u16.size < total:
        print(f"ERROR: X file too short: elements={x_u16.size}, required={total}")
        sys.exit(2)
    y_u16 = np.array(x_u16[:total], dtype=np.uint16, copy=True)

    # Prepare ranges derived from bf16 constants (as float32)
    max_f32 = bf16_from_bits_u16(np.array([BF16_MAX], dtype=np.uint16)).float()[0].item()
    min_norm_f32 = bf16_from_bits_u16(np.array([BF16_MIN_NORMAL], dtype=np.uint16)).float()[0].item()

    # Helper to write a row from a torch tensor of dtype bfloat16
    def write_row_pos(pos: int, t_bf16: torch.Tensor):
        """Write row at absolute row index 0..63 in the output file."""
        assert t_bf16.dtype == torch.bfloat16 and t_bf16.numel() == COLS
        start = pos * COLS
        end = start + COLS
        y_u16[start:end] = torch_bf16_to_uint16(t_bf16)

    # Rows 46..53 definitions (overwrite those rows in Y)
    # Row 46: uniform [0.0, 10.0]
    gen46 = torch.Generator(device='cpu').manual_seed(SEED + 46)
    r46 = torch.empty(COLS, dtype=torch.float32).uniform_(0.0, 10.0, generator=gen46).to(torch.bfloat16)
    write_row_pos(46, r46)

    # Row 47: uniform [-10.0, 0.0]
    gen47 = torch.Generator(device='cpu').manual_seed(SEED + 47)
    r47 = torch.empty(COLS, dtype=torch.float32).uniform_(-10.0, 0.0, generator=gen47).to(torch.bfloat16)
    write_row_pos(47, r47)

    # Row 48: uniform [BF16_MAX/2, BF16_MAX]
    gen48 = torch.Generator(device='cpu').manual_seed(SEED + 48)
    r48 = torch.empty(COLS, dtype=torch.float32).uniform_(float(max_f32/2.0), float(max_f32), generator=gen48).to(torch.bfloat16)
    write_row_pos(48, r48)

    # Row 49: uniform [BF16_MIN_NORMAL, BF16_MIN_NORMAL*10]
    gen49 = torch.Generator(device='cpu').manual_seed(SEED + 49)
    r49 = torch.empty(COLS, dtype=torch.float32).uniform_(float(min_norm_f32), float(min_norm_f32*10.0), generator=gen49).to(torch.bfloat16)
    write_row_pos(49, r49)

    # Row 50: random with injected -Inf
    gen50 = torch.Generator(device='cpu').manual_seed(SEED + 50)
    base50 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0, generator=gen50).to(torch.bfloat16)
    bits50 = torch_bf16_to_uint16(base50)
    inj_ninf = [11, 79, 143, 211, 307, 419, 523, 647, 701, 757]
    for p in inj_ninf:
        if p < COLS:
            bits50[p] = BF16_INF_NEG
    write_row_pos(50, bf16_from_bits_u16(bits50))

    # Row 51: random with injected +Inf
    gen51 = torch.Generator(device='cpu').manual_seed(SEED + 51)
    base51 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0, generator=gen51).to(torch.bfloat16)
    bits51 = torch_bf16_to_uint16(base51)
    inj_pinf = [7, 63, 127, 193, 271, 349, 509, 569, 631, 743]
    for p in inj_pinf:
        if p < COLS:
            bits51[p] = BF16_INF_POS
    write_row_pos(51, bf16_from_bits_u16(bits51))

    # Row 52: random with injected NaN
    gen52 = torch.Generator(device='cpu').manual_seed(SEED + 52)
    base52 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0, generator=gen52).to(torch.bfloat16)
    bits52 = torch_bf16_to_uint16(base52)
    inj_nan = [5, 41, 97, 157, 233, 307, 389, 457, 613, 727]
    for p in inj_nan:
        if p < COLS:
            bits52[p] = BF16_QNAN
    write_row_pos(52, bf16_from_bits_u16(bits52))

    # Row 53: random with injected +Inf (for 0 * Inf, etc.)
    gen53 = torch.Generator(device='cpu').manual_seed(SEED + 53)
    base53 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0, generator=gen53).to(torch.bfloat16)
    bits53 = torch_bf16_to_uint16(base53)
    inj_pinf2 = [3, 29, 73, 109, 181, 269, 347, 463, 557, 739]
    for p in inj_pinf2:
        if p < COLS:
            bits53[p] = BF16_INF_POS
    write_row_pos(53, bf16_from_bits_u16(bits53))

    # Save
    y_u16.tofile(Y_FILE)

    print(f"Saved: {Y_FILE}")
    print(f"Total elements written: {y_u16.size} (shape {ROWS} x {COLS})")

    # Optional: verify that rows other than 46..53 remain identical to X
    # and rows 46..53 differ (or match expected patterns). This is a light check.
    diff_mask = (y_u16 != x_u16[:total]).reshape(ROWS, COLS)
    changed_rows = [i for i in range(ROWS) if np.any(diff_mask[i])]
    print(f"Rows changed vs X: {changed_rows}")

    # Show a small head dump
    head = y_u16[:16].tolist()
    print("Head bf16 bits (hex):", [f"0x{v:04x}" for v in head])


if __name__ == '__main__':
    main()
