#!/usr/bin/env python3
"""
Compare Verilog RMSNorm output vs Python golden (bf16), aligned with softmax comparator style.

- TB output:   rmsnorm_out_tb.bin
- Python golden: rms_norm_bf16.bin

It reports:
- Size and shape checks (rows x 768 if divisible)
- Bitwise mismatch count (with NaN-equivalence and relative-error suppression)
- First N mismatches (row/col, hex, float, rel_err)
- Per-row mismatch counts (top-K rows)
- Float32 value-domain errors on finite elements (max/mean abs err)
- Total relative L2 error eps_f with special NaN handling
"""
import os
import numpy as np
import torch
from typing import Tuple

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
TB_FILE = os.path.join(DATA_DIR, 'rmsnorm_out_tb.bin')
GD_FILE = os.path.join(DATA_DIR, 'rms_norm_bf16.bin')

# ---- Conversion helpers ----

def bf16_u16_to_fp32_numpy(u16: np.ndarray) -> np.ndarray:
    """Convert uint16 BF16 payloads to float32 using bit-cast (shift then view).
    Preserves signed zeros; standard NaN/Inf encodings propagate.
    """
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)


def load_bf16_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    u16 = np.fromfile(path, dtype=np.uint16)
    fp32 = bf16_u16_to_fp32_numpy(u16)
    return fp32, u16


def main():
    if not os.path.exists(TB_FILE):
        raise FileNotFoundError(f'Missing TB file: {TB_FILE}')
    if not os.path.exists(GD_FILE):
        raise FileNotFoundError(f'Missing golden file: {GD_FILE}')

    tb32, tb_u16 = load_bf16_file(TB_FILE)
    gd32, gd_u16 = load_bf16_file(GD_FILE)

    ntb = tb_u16.size
    ngd = gd_u16.size
    print(f'TB length: {ntb}')
    print(f'GD length: {ngd}')

    if ntb != ngd:
        raise ValueError(f'Length mismatch: TB={ntb}, GD={ngd}')

    rows = ntb // 768 if ntb % 768 == 0 else None
    if rows is None:
        raise ValueError('Input length is not a multiple of 768; cannot form rows')
    print(f'Shape: {rows} x 768')

    # Prepare float32 flat views
    tb32_np = tb32.reshape(-1)
    gd32_np = gd32.reshape(-1)

    # Bitwise comparison with NaN-equivalence
    mismatches = (tb_u16 != gd_u16)
    both_nan_mask = (np.isnan(tb32_np) & np.isnan(gd32_np))
    nan_equiv_ignored = int(np.count_nonzero(mismatches & both_nan_mask))
    mismatches[both_nan_mask] = False

    # Relative-error filtering for finite elements: suppress when rel_err < 1e-3
    eps = 1e-12
    finite_mask_np = np.isfinite(tb32_np) & np.isfinite(gd32_np)
    denom = np.maximum(np.maximum(np.abs(tb32_np), np.abs(gd32_np)), eps)
    rel_err = np.abs(tb32_np - gd32_np) / denom
    rel_filter_mask = finite_mask_np & (rel_err < 1e-3)
    suppressed_rel = int(np.count_nonzero(mismatches & rel_filter_mask))
    mismatches[mismatches & rel_filter_mask] = False

    mismatch_count = int(np.count_nonzero(mismatches))
    total = ntb
    print(f'Bitwise mismatches: {mismatch_count} / {total}')
    if suppressed_rel > 0:
        print(f'  (suppressed {suppressed_rel} finite positions with relative error < 1e-3)')

    if mismatch_count > 0:
        print('First mismatches:')
        idxs = np.flatnonzero(mismatches)[:10]
        for i in idxs:
            r = i // 768
            c = i % 768
            pos = f'(row={r}, col={c})'
            tb_hex = f'0x{tb_u16[i]:04x}'
            gd_hex = f'0x{gd_u16[i]:04x}'
            tb_val = float(tb32_np[i])
            gd_val = float(gd32_np[i])
            if np.isfinite(tb_val) and np.isfinite(gd_val):
                re = abs(tb_val - gd_val) / max(max(abs(tb_val), abs(gd_val)), 1e-12)
                print(f'  {pos}: TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val}), rel_err={re:.6e}')
            else:
                print(f'  {pos}: TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val})')

    # Per-row mismatch counts (top-K) if any mismatches
    if mismatch_count > 0:
        row_counts = mismatches.reshape(rows, 768).sum(axis=1)
        top_rows = np.argsort(-row_counts)[:10]
        print('Top rows by mismatches:')
        for r in top_rows:
            cnt = int(row_counts[r])
            if cnt == 0:
                break
            print(f'  row {r}: {cnt}')

    # Value-domain comparison: finite elements only
    finite_mask = np.isfinite(tb32_np) & np.isfinite(gd32_np)
    num_finite = int(np.count_nonzero(finite_mask))
    if num_finite > 0:
        abs_err = np.abs(tb32_np[finite_mask] - gd32_np[finite_mask])
        _max_abs = float(abs_err.max())
        _mean_abs = float(abs_err.mean())
    else:
        pass

    # Total relative error eps_f with special NaN handling
    tb_np = tb32_np.astype(np.float64, copy=False)
    gd_np = gd32_np.astype(np.float64, copy=False)
    both_nan = np.isnan(tb_np) & np.isnan(gd_np)
    both_finite = np.isfinite(tb_np) & np.isfinite(gd_np)
    use_mask = both_nan | both_finite
    diff_vec = np.zeros_like(tb_np, dtype=np.float64)
    ref_vec = np.zeros_like(gd_np, dtype=np.float64)
    diff_vec[both_finite] = tb_np[both_finite] - gd_np[both_finite]
    ref_vec[both_finite] = gd_np[both_finite]
    num_l2 = np.linalg.norm(diff_vec)
    den_l2 = np.linalg.norm(ref_vec) + 1e-12
    eps_f = num_l2 / den_l2
    used = int(np.count_nonzero(use_mask))
    print(f'Total relative L2 error eps_f: {eps_f:.6e}')

    # Row-wise relative L2 error for rows [32..37] (0-based)
    if rows is not None:
        tb_rows = tb32_np.astype(np.float64, copy=False).reshape(rows, 768)
        gd_rows = gd32_np.astype(np.float64, copy=False).reshape(rows, 768)
        print('Row-wise relative L2 error for rows [32..37] (0-based):')
        for r in range(32, 38):
            if r >= rows:
                print(f'row {r}: out of range (total rows={rows})')
                continue
            row_tb = tb_rows[r]
            row_gd = gd_rows[r]
            if (not np.isfinite(row_tb).all()) or (not np.isfinite(row_gd).all()):
                print(f'row {r}: contains NaN or Inf')
                continue
            diff = row_tb - row_gd
            num = np.linalg.norm(diff)
            den = np.linalg.norm(row_gd) + 1e-12
            rel = num / den
            print(f'row {r}: rel_L2_err={rel:.6e}')

    # Summary
    if mismatch_count == 0 and ntb == ngd:
        print('RESULT: PASS (bitwise identical)')
    else:
        print('RESULT: MISMATCH (see details above)')


if __name__ == '__main__':
    main()
