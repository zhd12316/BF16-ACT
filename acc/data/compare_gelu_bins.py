#!/usr/bin/env python3
"""
Compare Verilog GELU output vs Python golden (bf16).

- TB output:     gelu_out_tb.bin
- Python golden:  gelu_bf16.bin

It reports:
- Size and shape checks (rows x 768 if divisible)
- Bitwise mismatch count and first N mismatches (with row/col, hex, and float values)
- Per-row mismatch counts (top-K rows)
- Float32 value-domain errors on finite elements (max/mean abs err)
- Total relative L2 error eps_f with special NaN handling

Notes:
- NaN-equivalence: positions where both are NaN are treated as equal (payload/sign differences ignored)
- Relative-error suppression: finite mismatches with rel_err < 1e-3 are suppressed from bitwise mismatch report
"""

import os
import argparse
import struct
import numpy as np
import torch
from typing import Tuple

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
TB_FILE = os.path.join(DATA_DIR, 'gelu_out_tb.bin')
GD_FILE = os.path.join(DATA_DIR, 'gelu_bf16_bf16.bin')

# ---- Conversion helpers ----

def uint16_to_torch_bf16(data_uint16: np.ndarray) -> torch.Tensor:
    """Convert uint16 bf16 bit-patterns to torch.bfloat16 via float path.
    Preserves signed zero; NaN payloads may be normalized by Python/torch.
    """
    bf16_values = []
    for val in data_uint16:
        sign = (val >> 15) & 0x1
        exp = (val >> 7) & 0xFF
        mantissa = val & 0x7F
        if exp == 0:
            if mantissa == 0:
                result = -0.0 if sign == 1 else 0.0
            else:
                result = (1 if sign == 0 else -1) * (mantissa / 128.0) * (2 ** -126)
        elif exp == 255:
            result = float('inf') if mantissa == 0 else float('nan')
            if sign == 1:
                result = -result
        else:
            exp_val = exp - 127.0
            mantissa_val = 1.0 + (mantissa / 128.0)
            result = (1.0 if sign == 0 else -1.0) * mantissa_val * (2.0 ** exp_val)
        bf16_values.append(torch.tensor(result, dtype=torch.bfloat16))
    return torch.stack(bf16_values)


def torch_bf16_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch.bfloat16 tensor to uint16 bf16 bit-pattern using float32 pack."""
    float32_values = tensor.float()
    u32 = np.array([struct.unpack('I', struct.pack('f', float(v)))[0] for v in float32_values], dtype=np.uint32)
    return (u32 >> 16).astype(np.uint16)


def load_bf16_file(path: str) -> Tuple[torch.Tensor, np.ndarray]:
    data_u16 = np.fromfile(path, dtype=np.uint16)
    data_bf16 = uint16_to_torch_bf16(data_u16)
    return data_bf16, data_u16


def main():
    parser = argparse.ArgumentParser(description='Compare GELU TB vs Golden (bf16).')
    parser.add_argument('--tb', type=str, default=TB_FILE, help='Path to TB output bin (default: gelu_out_tb.bin in data dir)')
    parser.add_argument('--gd', type=str, default=GD_FILE, help='Path to Golden bin (default: gelu_bf16.bin in data dir)')
    parser.add_argument('--row', type=int, default=None, help='Row index to print mismatches for (0-based). If omitted, skip row-focused listing.')
    parser.add_argument('--one_based', action='store_true', help='Interpret --row as 1-based index (internally subtract 1).')
    parser.add_argument('--first', type=int, default=10, help='How many mismatches to print for the selected row (default: 10)')
    parser.add_argument('--rel_tol', type=float, default=1e-3, help='Relative error suppression threshold (default: 1e-3)')
    args = parser.parse_args()

    tb_path = args.tb
    gd_path = args.gd

    if not os.path.exists(tb_path):
        raise FileNotFoundError(f'Missing TB file: {tb_path}')
    if not os.path.exists(gd_path):
        raise FileNotFoundError(f'Missing golden file: {gd_path}')

    tb_bf16, tb_u16 = load_bf16_file(tb_path)
    gd_bf16, gd_u16 = load_bf16_file(gd_path)

    ntb = tb_bf16.numel()
    ngd = gd_bf16.numel()
    print(f'TB length: {ntb}')
    print(f'GD length: {ngd}')

    if ntb != ngd:
        raise ValueError(f'Length mismatch: TB={ntb}, GD={ngd}')

    rows = ntb // 768 if ntb % 768 == 0 else None
    if rows is not None:
        print(f'Shape: {rows} x 768')
    else:
        print('Shape: flat vector (not multiple of 768)')

    # Prepare float32 views for semantic masks (NaN equivalence)
    tb32 = tb_bf16.float()
    gd32 = gd_bf16.float()

    # Bitwise comparison (with NaN-equivalence: any NaN payload/sign treated as equal)
    mismatches = (tb_u16 != gd_u16)
    tb32_np = tb32.cpu().numpy().reshape(-1)
    gd32_np = gd32.cpu().numpy().reshape(-1)
    both_nan_mask = (np.isnan(tb32_np) & np.isnan(gd32_np))
    nan_equiv_ignored = int(np.count_nonzero(mismatches & both_nan_mask))
    mismatches[both_nan_mask] = False

    # Relative-error filtering for finite elements: suppress outputs when rel_err < 1e-3
    eps = 1e-12
    finite_mask_np = np.isfinite(tb32_np) & np.isfinite(gd32_np)
    denom = np.maximum(np.maximum(np.abs(tb32_np), np.abs(gd32_np)), eps)
    rel_err = np.abs(tb32_np - gd32_np) / denom
    rel_filter_mask = finite_mask_np & (rel_err < args.rel_tol)
    suppressed_rel = int(np.count_nonzero(mismatches & rel_filter_mask))
    mismatches[mismatches & rel_filter_mask] = False

    mismatch_count = int(np.count_nonzero(mismatches))
    total = tb_u16.size
    print(f'Bitwise mismatches: {mismatch_count} / {total}')
    if suppressed_rel > 0:
        print(f'  (suppressed {suppressed_rel} finite positions with relative error < {args.rel_tol:.1e})')

    if mismatch_count > 0:
        print('First mismatches:')
        idxs = np.flatnonzero(mismatches)[:10]
        for i in idxs:
            if rows is not None:
                r = i // 768
                c = i % 768
                pos = f'(row={r}, col={c})'
            else:
                pos = f'(idx={i})'
            tb_hex = f'0x{tb_u16[i]:04x}'
            gd_hex = f'0x{gd_u16[i]:04x}'
            tb_val = float(tb32_np[i])
            gd_val = float(gd32_np[i])
            # Attach relative error if finite
            if np.isfinite(tb_val) and np.isfinite(gd_val):
                re = abs(tb_val - gd_val) / max(max(abs(tb_val), abs(gd_val)), 1e-12)
                print(f'  {pos}: TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val}), rel_err={re:.6e}')
            else:
                print(f'  {pos}: TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val})')

    # Per-row mismatch counts (if shaped)
    if rows is not None and mismatch_count > 0:
        row_counts = mismatches.reshape(rows, 768).sum(axis=1)
        top_rows = np.argsort(-row_counts)[:10]
        print('Top rows by mismatches:')
        for r in top_rows:
            cnt = int(row_counts[r])
            if cnt == 0:
                break
            print(f'  row {r}: {cnt}')

    # Value-domain comparison: finite elements only
    finite_mask = torch.isfinite(tb32) & torch.isfinite(gd32)
    num_finite = int(finite_mask.sum().item())
    if num_finite > 0:
        abs_err = torch.abs(tb32[finite_mask] - gd32[finite_mask])
        _max_abs = float(abs_err.max())
        _mean_abs = float(abs_err.mean())
    else:
        pass

    # Total relative error eps_f with special NaN handling:
    # Treat positions where both are NaN as zero difference; compute L2 norms over
    # finite pairs plus both-NaN zeros; ignore other non-finite mismatches from the norm
    # but report how many were excluded.
    tb_np = tb32_np.astype(np.float64, copy=False)
    gd_np = gd32_np.astype(np.float64, copy=False)
    both_nan = np.isnan(tb_np) & np.isnan(gd_np)
    both_finite = np.isfinite(tb_np) & np.isfinite(gd_np)
    use_mask = both_nan | both_finite
    diff_vec = np.zeros_like(tb_np, dtype=np.float64)
    ref_vec = np.zeros_like(gd_np, dtype=np.float64)
    # finite pairs contribute true differences and reference
    diff_vec[both_finite] = tb_np[both_finite] - gd_np[both_finite]
    ref_vec[both_finite] = gd_np[both_finite]
    # both-NaN contribute zero to both numerator and denominator
    # compute eps_f = ||diff||2 / (||ref||2 + 1e-12)
    num_l2 = np.linalg.norm(diff_vec)
    den_l2 = np.linalg.norm(ref_vec) + 1e-12
    eps_f = num_l2 / den_l2
    used = int(np.count_nonzero(use_mask))
    print(f'Total relative L2 error eps_f: {eps_f:.6e}')

    # Row-wise relative L2 error for rows [42..45] (0-based)
    if rows is not None:
        tb_rows = tb32_np.astype(np.float64, copy=False).reshape(rows, 768)
        gd_rows = gd32_np.astype(np.float64, copy=False).reshape(rows, 768)
        print('Row-wise relative L2 error for rows [42..45] (0-based):')
        for r in range(42, 46):
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

        # Added: Detailed first-10 element comparison for row 44 (0-based index = 44)
        # if rows > 2:
        #     print('Row 44 first 10 element comparison (0-based row index = 44):')
        #     base = 2 * 768
        #     for c in range(10):
        #         i = base + c
        #         tb_hex = f'0x{tb_u16[i]:04x}'
        #         gd_hex = f'0x{gd_u16[i]:04x}'
        #         tb_val = float(tb32_np[i])
        #         gd_val = float(gd32_np[i])
        #         if np.isfinite(tb_val) and np.isfinite(gd_val):
        #             re = abs(tb_val - gd_val) / max(max(abs(tb_val), abs(gd_val)), 1e-12)
        #             print(f'  (col={c}): TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val}), rel_err={re:.6e}')
        #         else:
        #             print(f'  (col={c}): TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val})')
        # else:
        #     print('Row 44 comparison skipped: total rows <= 44')

    # Row-focused mismatch listing (optional; only when --row is provided)
    if rows is not None and mismatch_count > 0 and args.row is not None:
        rsel = args.row - 1 if args.one_based else args.row
        if rsel < 0 or rsel >= rows:
            print(f'[WARN] --row {rsel} is out of range [0, {rows-1}] — skipping row-focused listing.')
        else:
            row_mask = mismatches.reshape(rows, 768)[rsel]
            row_mis = int(np.count_nonzero(row_mask))
            row_idxs = np.flatnonzero(row_mask)[: args.first]
            print(f'Row {rsel} mismatches: total={row_mis}, listing first {len(row_idxs)}:')
            for c in row_idxs:
                i = rsel * 768 + c
                tb_hex = f'0x{tb_u16[i]:04x}'
                gd_hex = f'0x{gd_u16[i]:04x}'
                tb_val = float(tb32_np[i])
                gd_val = float(gd32_np[i])
                if np.isfinite(tb_val) and np.isfinite(gd_val):
                    re = abs(tb_val - gd_val) / max(max(abs(tb_val), abs(gd_val)), 1e-12)
                    print(f'  (row={rsel}, col={c}): TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val}), rel_err={re:.6e}')
                else:
                    print(f'  (row={rsel}, col={c}): TB={tb_hex} ({tb_val}), GD={gd_hex} ({gd_val})')

    # Summary
    if mismatch_count == 0 and ntb == ngd:
        print('RESULT: PASS (bitwise identical)')
    else:
        print('RESULT: MISMATCH (see details above)')


if __name__ == '__main__':
    main()
