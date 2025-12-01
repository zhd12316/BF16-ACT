#!/usr/bin/env python3
"""
Quick verifier for X_test_tensor_bf16.bin and Y_test_tensor_bf16.bin.
- Checks sizes match expected shapes (X: 64x768, Y: 8x768 per current spec)
- Performs targeted bit-pattern and value checks on key rows

Output: PASS/FAIL per check for fast confidence.
"""
import os
import sys
import numpy as np
import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
X_PATH = os.path.join(DATA_DIR, 'X_test_tensor_bf16.bin')
Y_PATH = os.path.join(DATA_DIR, 'Y_test_tensor_bf16.bin')

ROWS_X = 64
ROWS_Y = 64  # Y is now full 64 rows (rows 46..53 are overridden)
COLS = 768

# bf16 special constants (bit patterns)
BF16_POS_ZERO   = 0x0000
BF16_NEG_ZERO   = 0x8000
BF16_INF_POS    = 0x7F80
BF16_INF_NEG    = 0xFF80
BF16_QNAN       = 0x7FC0
BF16_MAX        = 0x7F7F
BF16_MAX_NEG    = 0xFF7F
BF16_MIN_NORMAL = 0x0080  # already used elsewhere; re-affirm
BF16_MIN_NORMAL_NEG = 0x8080
BF16_SUB_MAX    = 0x007F
BF16_SUB_MAX_NEG= 0x807F


def torch_bf16_to_uint16(t: torch.Tensor) -> np.ndarray:
    f32 = t.float().cpu().numpy().astype(np.float32, copy=False)
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)

def bf16_from_bits_u16(bits_u16: np.ndarray) -> torch.Tensor:
    bits_u16 = np.asarray(bits_u16, dtype=np.uint16)
    bits_u32 = bits_u16.astype(np.uint32) << 16
    f32 = bits_u32.view(np.float32)
    return torch.from_numpy(f32).to(torch.bfloat16)


def read_u16(path: str) -> np.ndarray:
    if not os.path.exists(path):
        print(f'ERROR: file not found: {path}')
        sys.exit(1)
    return np.fromfile(path, dtype=np.uint16)


def check_all_equal_bits(row_bits: np.ndarray, target_bits: int) -> bool:
    return bool(np.all(row_bits == np.uint16(target_bits)))

def check_alternating_zero_sign(row_bits: np.ndarray) -> bool:
    if row_bits.size < 2:
        return False
    even_ok = np.all(row_bits[0::2] == np.uint16(BF16_POS_ZERO))
    odd_ok = np.all(row_bits[1::2] == np.uint16(BF16_NEG_ZERO))
    return bool(even_ok and odd_ok)

def check_injections(row_bits: np.ndarray, positions: list[int], cycle: list[int]) -> bool:
    """
    Check that row_bits at given positions match a repeating cycle of expectations.
    Special handling: if expected is BF16_QNAN, accept any NaN payload/sign
    (exp=0xFF and mantissa!=0), since NaN payload/sign can change across casts.
    """
    ok = True
    for i, p in enumerate(positions):
        if p < 0 or p >= row_bits.size:
            ok = False
            print(f'  FAIL: injection index out of bounds: {p}')
            continue
        expect = cycle[i % len(cycle)]
        got = int(row_bits[p])
        if expect == BF16_QNAN:
            # Any NaN: exponent all ones (mask 0x7F80) and mantissa != 0
            is_nan = ((got & 0x7F80) == 0x7F80) and ((got & 0x007F) != 0)
            if not is_nan:
                ok = False
                print(f'  FAIL: pos {p} expected NaN, got 0x{got:04x}')
        else:
            if got != expect:
                ok = False
                print(f'  FAIL: pos {p} expected 0x{expect:04x}, got 0x{got:04x}')
    return ok

def check_monotonic_increasing(row_bits: np.ndarray) -> bool:
    # Convert to float for monotonicity check (bf16 -> f32)
    vals = bf16_from_bits_u16(row_bits).float().numpy()
    return bool(np.all(np.diff(vals) >= -1e-7))  # allow tiny equal/rounding

def check_monotonic_decreasing(row_bits: np.ndarray) -> bool:
    vals = bf16_from_bits_u16(row_bits).float().numpy()
    return bool(np.all(np.diff(vals) <= 1e-7))

def check_range_bits_as_float(row_bits: np.ndarray, low: float, high: float) -> bool:
    vals = bf16_from_bits_u16(row_bits).float().numpy()
    return bool(np.all(vals >= low - 1e-6) and np.all(vals <= high + 1e-6))

def is_nan_bits(v: int) -> bool:
    return ((v & 0x7F80) == 0x7F80) and ((v & 0x007F) != 0)

def is_subnormal_pos(v: int) -> bool:
    return (v & 0x7F80) == 0 and (v & 0x007F) != 0 and (v & 0x8000) == 0

def is_subnormal_neg(v: int) -> bool:
    return (v & 0x7F80) == 0 and (v & 0x007F) != 0 and (v & 0x8000) != 0

def count_specials(row_bits: np.ndarray) -> dict:
    vals = row_bits.astype(np.uint16).astype(np.int32)
    d = {
        'pos_zero': int(np.sum(vals == BF16_POS_ZERO)),
        'neg_zero': int(np.sum(vals == BF16_NEG_ZERO)),
        'pos_inf': int(np.sum(vals == BF16_INF_POS)),
        'neg_inf': int(np.sum(vals == BF16_INF_NEG)),
        'nan': int(np.sum([is_nan_bits(int(v)) for v in vals])),
        'sub_pos': int(np.sum([is_subnormal_pos(int(v)) for v in vals])),
        'sub_neg': int(np.sum([is_subnormal_neg(int(v)) for v in vals])),
    }
    return d

def row_summary(name: str, row_bits: np.ndarray):
    stats = count_specials(row_bits)
    vals = bf16_from_bits_u16(row_bits).float().numpy()
    finite = vals[np.isfinite(vals)]
    fmin = float(np.min(finite)) if finite.size else float('nan')
    fmax = float(np.max(finite)) if finite.size else float('nan')
    print(f"{name}: nan={stats['nan']}, +inf={stats['pos_inf']}, -inf={stats['neg_inf']}, +0={stats['pos_zero']}, -0={stats['neg_zero']}, sub+={stats['sub_pos']}, sub-={stats['sub_neg']}, finite[min,max]=[{fmin:.6e},{fmax:.6e}]")


def main():
    # Read X and Y
    x = read_u16(X_PATH)
    y = read_u16(Y_PATH)

    exp_len_x = ROWS_X * COLS
    exp_len_y = ROWS_Y * COLS

    print('== Size checks ==')
    print(f'X length: {x.size}, expected: {exp_len_x}  ->', 'PASS' if x.size == exp_len_x else 'FAIL')
    print(f'Y length: {y.size}, expected: {exp_len_y}  ->', 'PASS' if y.size == exp_len_y else 'FAIL')
    if x.size != exp_len_x or y.size != exp_len_y:
        print('Early exit due to size mismatch.')
        sys.exit(2)

    # Helper to get row slice from flat array
    def row_bits(arr: np.ndarray, row_idx: int) -> np.ndarray:
        s = row_idx * COLS
        e = s + COLS
        return arr[s:e]

    print('\n== X targeted checks ==')
    # Row 0: all +0.0
    x_r0 = row_bits(x, 0)
    print('X row 0 all +0.0 ->', 'PASS' if check_all_equal_bits(x_r0, BF16_POS_ZERO) else 'FAIL')

    # Row 1: all 1.0
    ones_bits = torch_bf16_to_uint16(torch.full((COLS,), 1.0, dtype=torch.bfloat16))
    x_r1 = row_bits(x, 1)
    print('X row 1 all 1.0 ->', 'PASS' if np.all(x_r1 == ones_bits) else 'FAIL')

    # Row 11: alternating +0.0 / -0.0
    x_r11 = row_bits(x, 11)
    print('X row 11 +/-0.0 alternating ->', 'PASS' if check_alternating_zero_sign(x_r11) else 'FAIL')

    # Row 20: large positive increasing sequence (monotonic)
    x_r20 = row_bits(x, 20)
    print('X row 20 monotonic increasing ->', 'PASS' if check_monotonic_increasing(x_r20) else 'FAIL')

    # Row 24: two +Inf at 1/3 and 2/3, others 0
    x_r24 = row_bits(x, 24)
    i1, i2 = COLS // 3, (2 * COLS) // 3
    cond = (x_r24[i1] == BF16_INF_POS) and (x_r24[i2] == BF16_INF_POS)
    cond = cond and np.all(np.delete(x_r24, [i1, i2]) == BF16_POS_ZERO)
    print('X row 24 two +Inf at 1/3 & 2/3 ->', 'PASS' if cond else 'FAIL')

    # Row 41: injected +Inf/-Inf/NaN at specific positions
    inj_pos3 = [7, 49, 97, 149, 223, 307, 401, 503, 607, 701]
    cycle = [BF16_INF_POS, BF16_INF_NEG, BF16_QNAN]
    x_r41 = row_bits(x, 41)
    print('X row 41 injected specials ->', 'PASS' if check_injections(x_r41, inj_pos3, cycle) else 'FAIL')

    print('\n== X all rows summary ==')
    for ri in range(ROWS_X):
        xb = row_bits(x, ri)
        row_summary(f'X row {ri:02d}', xb)

    # Optional per-row assertions (broad coverage)
    print('\n== X additional assertions ==')
    checks = []
    # 0..8 fixed patterns
    checks.append(('X r0 all +0', check_all_equal_bits(row_bits(x,0), BF16_POS_ZERO)))
    ones_bits = torch_bf16_to_uint16(torch.full((COLS,), 1.0, dtype=torch.bfloat16))
    neg1_bits = torch_bf16_to_uint16(torch.full((COLS,), -1.0, dtype=torch.bfloat16))
    checks.append(('X r1 all 1.0', bool(np.all(row_bits(x,1) == ones_bits))))
    checks.append(('X r2 all -1.0', bool(np.all(row_bits(x,2) == neg1_bits))))
    checks.append(('X r3 all +BF16_MAX', check_all_equal_bits(row_bits(x,3), BF16_MAX)))
    checks.append(('X r4 all -BF16_MAX', check_all_equal_bits(row_bits(x,4), BF16_MAX_NEG)))
    checks.append(('X r5 +BF16_MIN_NORMAL', check_all_equal_bits(row_bits(x,5), BF16_MIN_NORMAL)))
    checks.append(('X r6 -BF16_MIN_NORMAL', check_all_equal_bits(row_bits(x,6), BF16_MIN_NORMAL_NEG)))
    checks.append(('X r7 +subnormal max', check_all_equal_bits(row_bits(x,7), BF16_SUB_MAX)))
    checks.append(('X r8 -subnormal max', check_all_equal_bits(row_bits(x,8), BF16_SUB_MAX_NEG)))

    # 11 alternating signed zero
    checks.append(('X r11 alt +/-0', check_alternating_zero_sign(row_bits(x,11))))

    # 12..15 ranges
    checks.append(('X r12 in [-10,10]', check_range_bits_as_float(row_bits(x,12), -10.0, 10.0)))
    checks.append(('X r13 in [-1000,1000]', check_range_bits_as_float(row_bits(x,13), -1000.0, 1000.0)))
    checks.append(('X r14 in [-0.1,0.1]', check_range_bits_as_float(row_bits(x,14), -0.1, 0.1)))
    checks.append(('X r15 in [0,1]', check_range_bits_as_float(row_bits(x,15), 0.0, 1.0)))

    # 16 variance > 0 and contains both signs
    vals16 = bf16_from_bits_u16(row_bits(x,16)).float().numpy()
    checks.append(('X r16 mixed signs', bool(np.any(vals16>0) and np.any(vals16<0))))

    # 17 injections (+Inf / NaN alternating)
    inj_pos17 = [0, 17, 31, 63, 127, 255, 511, 767]
    checks.append(('X r17 injected +Inf/NaN', check_injections(row_bits(x,17), inj_pos17, [BF16_INF_POS, BF16_QNAN])))

    # 18 subnormal injections 0x0001 / 0x8001
    def check_sub_inj(bits):
        pos = [0, 17, 31, 63, 127, 255, 511, 767]
        ok = True
        for k,p in enumerate(pos):
            if p>=bits.size: continue
            got = int(bits[p])
            expect = 0x0001 if (k%2==0) else 0x8001
            if got != expect:
                ok=False; print(f'  FAIL: r18 pos {p} expect 0x{expect:04x}, got 0x{got:04x}')
        return ok
    checks.append(('X r18 subnormal injections', check_sub_inj(row_bits(x,18))))

    # 19 large positive range [0, BF16_MAX/2]
    max_f32 = bf16_from_bits_u16(np.array([BF16_MAX], dtype=np.uint16)).float()[0].item()
    checks.append(('X r19 in [0,bfmax/2]', check_range_bits_as_float(row_bits(x,19), 0.0, max_f32/2.0)))

    # 20 inc, 21 dec
    checks.append(('X r20 monotonic inc', check_monotonic_increasing(row_bits(x,20))))
    checks.append(('X r21 monotonic dec', check_monotonic_decreasing(row_bits(x,21))))

    # 22 single 10.0 at mid
    mid = COLS//2
    b22 = row_bits(x,22)
    v22 = bf16_from_bits_u16(b22).float().numpy()
    checks.append(('X r22 single 10 at mid', bool(np.isclose(v22[mid], 10.0, atol=1e-2) and np.all(np.delete(v22, mid)==0.0))))

    # 23 single +Inf at mid
    b23 = row_bits(x,23)
    checks.append(('X r23 single +Inf at mid', bool(b23[mid]==BF16_INF_POS and np.all(np.delete(b23,mid)==BF16_POS_ZERO))))

    # 24 two +Inf
    i1,i2 = COLS//3,(2*COLS)//3
    b24=row_bits(x,24)
    ok24 = bool(b24[i1]==BF16_INF_POS and b24[i2]==BF16_INF_POS and np.all(np.delete(b24,[i1,i2])==BF16_POS_ZERO))
    checks.append(('X r24 two +Inf', ok24))

    # 25 single NaN at mid
    b25=row_bits(x,25)
    checks.append(('X r25 NaN at mid', bool(is_nan_bits(int(b25[mid])))))

    # 26 alt ±1
    b26=row_bits(x,26)
    exp26 = torch_bf16_to_uint16(torch.tensor([1.0,-1.0]*((COLS+1)//2), dtype=torch.bfloat16)[:COLS])
    checks.append(('X r26 alt ±1', bool(np.all(b26==exp26))))

    # 27 near-constant small var (monotonic inc)
    checks.append(('X r27 monotonic inc', check_monotonic_increasing(row_bits(x,27))))

    # 28 injections (+Inf/NaN)
    inj_pos28 = [0, 29, 57, 101, 211, 333, 455, 699]
    checks.append(('X r28 injected +Inf/NaN', check_injections(row_bits(x,28), inj_pos28, [BF16_INF_POS, BF16_QNAN])))

    # 29 eps stepped non-decreasing
    checks.append(('X r29 non-decreasing', check_monotonic_increasing(row_bits(x,29))))

    # 30,31 ranges
    checks.append(('X r30 in [0,bfmax/10]', check_range_bits_as_float(row_bits(x,30), 0.0, max_f32/10.0)))
    min_norm_f32 = bf16_from_bits_u16(np.array([BF16_MIN_NORMAL], dtype=np.uint16)).float()[0].item()
    checks.append(('X r31 in [min_norm,10*min_norm]', check_range_bits_as_float(row_bits(x,31), min_norm_f32, min_norm_f32*10.0)))

    # 32 alt ±1, 33 inc
    checks.append(('X r32 alt ±1', bool(np.all(row_bits(x,32)==exp26))))
    checks.append(('X r33 monotonic inc', check_monotonic_increasing(row_bits(x,33))))

    # 34 injections (+Inf/NaN)
    inj_pos34 = [5, 23, 57, 89, 233, 377, 521, 733]
    checks.append(('X r34 injected +Inf/NaN', check_injections(row_bits(x,34), inj_pos34, [BF16_INF_POS, BF16_QNAN])))

    # 35 non-decreasing, 36/37 ranges
    checks.append(('X r35 non-decreasing', check_monotonic_increasing(row_bits(x,35))))
    checks.append(('X r36 in [0,bfmax/10]', check_range_bits_as_float(row_bits(x,36), 0.0, max_f32/10.0)))
    checks.append(('X r37 in [min_norm,10*min_norm]', check_range_bits_as_float(row_bits(x,37), min_norm_f32, min_norm_f32*10.0)))

    # 38/39 inc, 40 dec, 41 injections
    checks.append(('X r38 monotonic inc', check_monotonic_increasing(row_bits(x,38))))
    checks.append(('X r39 monotonic inc', check_monotonic_increasing(row_bits(x,39))))
    checks.append(('X r40 monotonic dec', check_monotonic_decreasing(row_bits(x,40))))
    inj_pos41 = [7, 49, 97, 149, 223, 307, 401, 503, 607, 701]
    checks.append(('X r41 injected +Inf/-Inf/NaN', check_injections(row_bits(x,41), inj_pos41, [BF16_INF_POS, BF16_INF_NEG, BF16_QNAN])))

    # 42..45 similar patterns
    checks.append(('X r42 monotonic inc', check_monotonic_increasing(row_bits(x,42))))
    checks.append(('X r43 monotonic inc', check_monotonic_increasing(row_bits(x,43))))
    checks.append(('X r44 monotonic dec', check_monotonic_decreasing(row_bits(x,44))))
    inj_pos45 = [3, 37, 83, 131, 197, 269, 347, 419, 587, 659, 731]
    checks.append(('X r45 injected +Inf/-Inf/NaN', check_injections(row_bits(x,45), inj_pos45, [BF16_INF_POS, BF16_INF_NEG, BF16_QNAN])))

    # 46..49 ranges
    checks.append(('X r46 in [0,10]', check_range_bits_as_float(row_bits(x,46), 0.0, 10.0)))
    checks.append(('X r47 in [-10,0]', check_range_bits_as_float(row_bits(x,47), -10.0, 0.0)))
    checks.append(('X r48 in [bfmax/2,bfmax]', check_range_bits_as_float(row_bits(x,48), max_f32/2.0, max_f32)))
    checks.append(('X r49 in [min_norm,10*min_norm]', check_range_bits_as_float(row_bits(x,49), min_norm_f32, min_norm_f32*10.0)))

    # 50..53 injections (+Inf, -Inf, NaN, +0)
    inj_pos_pinf = [11, 79, 143, 211, 307, 419, 523, 647, 701, 757]
    inj_pos_ninf = [7, 63, 127, 193, 271, 349, 509, 569, 631, 743]
    inj_pos_nan = [5, 41, 97, 157, 233, 307, 389, 457, 613, 727]
    inj_pos_zero= [3, 29, 73, 109, 181, 269, 347, 463, 557, 739]
    checks.append(('X r50 +Inf injections', check_injections(row_bits(x,50), inj_pos_pinf, [BF16_INF_POS])))
    checks.append(('X r51 -Inf injections', check_injections(row_bits(x,51), inj_pos_ninf, [BF16_INF_NEG])))
    checks.append(('X r52 NaN injections', check_injections(row_bits(x,52), inj_pos_nan, [BF16_QNAN])))
    # r53 injected +0.0 at specific positions
    b53 = row_bits(x,53)
    ok53 = True
    for p in inj_pos_zero:
        if p<COLS and b53[p]!=BF16_POS_ZERO:
            print(f'  FAIL: X r53 pos {p} expected +0.0 got 0x{int(b53[p]):04x}')
            ok53=False
    checks.append(('X r53 +0 injections', ok53))

    # 54..63 generic checks
    checks.append(('X r54 >0 finite', bool(np.all(np.isfinite(bf16_from_bits_u16(row_bits(x,54)).float().numpy())) and np.all(bf16_from_bits_u16(row_bits(x,54)).float().numpy()>0))))
    vals55 = bf16_from_bits_u16(row_bits(x,55)).float().numpy()
    checks.append(('X r55 >=0 finite', bool(np.all(np.isfinite(vals55)) and np.all(vals55>=0))))
    checks.append(('X r56 in [-1e-3,1e-3]', check_range_bits_as_float(row_bits(x,56), -1e-3, 1e-3)))
    checks.append(('X r57 in [-10,10]', check_range_bits_as_float(row_bits(x,57), -10.0, 10.0)))
    # r58 alternating +MAX/-MAX
    b58=row_bits(x,58)
    exp58 = np.empty(COLS, dtype=np.uint16); exp58[0::2]=BF16_MAX; exp58[1::2]=BF16_MAX_NEG
    checks.append(('X r58 alt ±BF16_MAX', bool(np.all(b58==exp58))))
    checks.append(('X r59 monotonic inc', check_monotonic_increasing(row_bits(x,59))))
    checks.append(('X r60 monotonic inc', check_monotonic_increasing(row_bits(x,60))))
    # r61 pattern [sub+, +0, sub-, -0]
    pat61 = np.array([BF16_SUB_MAX, BF16_POS_ZERO, BF16_SUB_MAX_NEG, BF16_NEG_ZERO], dtype=np.uint16)
    b61=row_bits(x,61)
    ok61 = True
    for idx in range(COLS):
        if b61[idx] != pat61[idx%4]:
            ok61=False; print(f'  FAIL: X r61 idx {idx} expect 0x{int(pat61[idx%4]):04x} got 0x{int(b61[idx]):04x}'); break
    checks.append(('X r61 sub/zero cycle', ok61))
    checks.append(('X r62 in [-2048,2048]', check_range_bits_as_float(row_bits(x,62), -2048.0, 2048.0)))
    checks.append(('X r63 in [-1000,1000]', check_range_bits_as_float(row_bits(x,63), -1000.0, 1000.0)))

    for name, ok in checks:
        print(f'{name} ->', 'PASS' if ok else 'FAIL')

    print('\n== Y targeted checks ==')
    # Verify rows equal to X except 46..53
    diff_mask = (y != x).reshape(ROWS_Y, COLS)
    changed_rows = [i for i in range(ROWS_Y) if np.any(diff_mask[i])]
    print('Y changed rows vs X ->', changed_rows)
    expect_changed = list(range(46, 54))
    print('Y changed rows match [46..53] ->', 'PASS' if changed_rows == expect_changed else 'FAIL')

    # Row 46: uniform [0,10]
    print('Y row 46 in [0,10] ->', 'PASS' if check_range_bits_as_float(row_bits(y,46), 0.0, 10.0) else 'FAIL')
    # Row 47: uniform [-10,0]
    print('Y row 47 in [-10,0] ->', 'PASS' if check_range_bits_as_float(row_bits(y,47), -10.0, 0.0) else 'FAIL')
    # Row 48: [bfmax/2, bfmax]
    max_f32 = bf16_from_bits_u16(np.array([BF16_MAX], dtype=np.uint16)).float()[0].item()
    print('Y row 48 in [bfmax/2,bfmax] ->', 'PASS' if check_range_bits_as_float(row_bits(y,48), max_f32/2.0, max_f32) else 'FAIL')
    # Row 49: [min_norm, 10*min_norm]
    min_norm_f32 = bf16_from_bits_u16(np.array([BF16_MIN_NORMAL], dtype=np.uint16)).float()[0].item()
    print('Y row 49 in [min_norm,10*min_norm] ->', 'PASS' if check_range_bits_as_float(row_bits(y,49), min_norm_f32, min_norm_f32*10.0) else 'FAIL')
    # Row 50: injected -Inf
    inj_ninf = [7, 63, 127, 193, 271, 349, 509, 569, 631, 743]
    print('Y row 50 injected -Inf ->', 'PASS' if check_injections(row_bits(y,50), inj_ninf, [BF16_INF_NEG]) else 'FAIL')
    # Row 51: injected +Inf
    inj_pinf = [11, 79, 143, 211, 307, 419, 523, 647, 701, 757]
    print('Y row 51 injected +Inf ->', 'PASS' if check_injections(row_bits(y,51), inj_pinf, [BF16_INF_POS]) else 'FAIL')
    # Row 52: injected NaN
    inj_nan = [5, 41, 97, 157, 233, 307, 389, 457, 613, 727]
    print('Y row 52 injected NaN ->', 'PASS' if check_injections(row_bits(y,52), inj_nan, [BF16_QNAN]) else 'FAIL')
    # Row 53: injected +Inf
    inj_pinf2 = [3, 29, 73, 109, 181, 269, 347, 463, 557, 739]
    print('Y row 53 injected +Inf ->', 'PASS' if check_injections(row_bits(y,53), inj_pinf2, [BF16_INF_POS]) else 'FAIL')

    print('\n== Y all rows summary ==')
    for ri in range(ROWS_Y):
        yb = row_bits(y, ri)
        row_summary(f'Y row {ri:02d}', yb)

    print('\n== Y additional assertions ==')
    # Assert rows other than 46..53 are identical to X
    same_rows = [i for i in range(ROWS_Y) if i < 46 or i > 53]
    same_ok = True
    for i in same_rows:
        if not np.array_equal(row_bits(y,i), row_bits(x,i)):
            print(f'  FAIL: Y row {i} not equal to X row {i}')
            same_ok = False
            break
    print('Y rows equal to X outside 46..53 ->', 'PASS' if same_ok else 'FAIL')

    print('\nDone.')


if __name__ == '__main__':
    main()
