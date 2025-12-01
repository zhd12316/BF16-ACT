#!/usr/bin/env python3


import os
import numpy as np
import torch
from typing import List, Optional

# Constants
ROWS = 64           # total rows
COLS = 768          # columns per row
SEED = 12345        # deterministic seed for reproducibility
EPSILON_LN = 1e-6   # epsilon for layer/rms norm boundary tests
EPSILON_RMS = 1e-6  # epsilon for rms-specific boundary tests
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
OUT_FILE = os.path.join(DATA_DIR, 'X_test_tensor_bf16.bin')

# ---- bf16 helpers ----

def bf16_from_bits_u16(bits_u16: np.ndarray) -> torch.Tensor:

    bits_u16 = np.asarray(bits_u16, dtype=np.uint16)
    bits_u32 = bits_u16.astype(np.uint32) << 16
    f32 = bits_u32.view(np.float32)
    return torch.from_numpy(f32).to(torch.bfloat16)

def torch_bf16_to_uint16(t: torch.Tensor) -> np.ndarray:
    """Convert torch.bfloat16 tensor to uint16 bf16 bit patterns (vectorized)."""
    f32 = t.float().cpu().numpy().astype(np.float32, copy=False)
    u32 = f32.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16

# Bit patterns for common bf16 constants
BF16_POS_ZERO   = 0x0000
BF16_NEG_ZERO   = 0x8000
BF16_INF_POS    = 0x7F80
BF16_INF_NEG    = 0xFF80
BF16_QNAN       = 0x7FC0  # a canonical quiet NaN (sign doesn't matter)
BF16_MAX        = 0x7F7F  # exp=254, mantissa=0x7F, sign=0
BF16_MAX_NEG    = 0xFF7F  # sign=1, exp=254, mantissa=0x7F
BF16_MIN_NORMAL = 0x0080  # exp=1,  mantissa=0,    sign=0  => +2^-126
BF16_MIN_NORMAL_NEG = 0x8080
BF16_SUB_MAX    = 0x007F  # largest positive subnormal (exp=0, mantissa=0x7F)
BF16_SUB_MAX_NEG= 0x807F

# ---- row builders ----

def row_fill_float(value: float) -> torch.Tensor:
    """Create a row of length COLS filled with a float value, cast to bf16."""
    return torch.full((COLS,), value, dtype=torch.bfloat16)

def row_fill_bits(bit_pattern: int) -> torch.Tensor:
    """Create a row of length COLS filled with the given bf16 bit pattern."""
    bits = np.full((COLS,), np.uint16(bit_pattern), dtype=np.uint16)
    return bf16_from_bits_u16(bits)

def row_from_list(values: list) -> torch.Tensor:
    """Create a row by repeating a list of python floats to length COLS, as bf16."""
    base = torch.tensor(values, dtype=torch.bfloat16)
    # Repeat and trim/pad to COLS
    reps = (COLS + base.numel() - 1) // base.numel()
    out = base.repeat(reps)[:COLS]
    return out

def row_from_bits_list(bits_list: list[int]) -> torch.Tensor:
    """Create a row by repeating a list of bf16 bit patterns to length COLS."""
    arr = np.array(bits_list, dtype=np.uint16)
    reps = (COLS + arr.size - 1) // arr.size
    tiled = np.tile(arr, reps)[:COLS]
    return bf16_from_bits_u16(tiled)


def build_rows() -> list[torch.Tensor]:
    rows_opt: List[Optional[torch.Tensor]] = [None] * ROWS

    # 行 0: 全零 (+0.0)
    rows_opt[0] = row_fill_bits(BF16_POS_ZERO)

    # 行 1: 全一 (1.0)
    rows_opt[1] = row_fill_float(1.0)

    # 行 2: 全负一 (-1.0)
    rows_opt[2] = row_fill_float(-1.0)

    # 行 3: 全 BF16_MAX (最大正有限数)
    rows_opt[3] = row_fill_bits(BF16_MAX)

    # 行 4: 全 -BF16_MAX (最大负有限数)
    rows_opt[4] = row_fill_bits(BF16_MAX_NEG)

    # 行 5: 全 BF16_MIN_NORMAL (最小正正规数)
    rows_opt[5] = row_fill_bits(BF16_MIN_NORMAL)

    # 行 6: 全 -BF16_MIN_NORMAL (最小负正规数)
    rows_opt[6] = row_fill_bits(BF16_MIN_NORMAL_NEG)

    # 行 7: 全正次正规数（使用最大的正次正规数）
    rows_opt[7] = row_fill_bits(BF16_SUB_MAX)

    # 行 8: 全负次正规数（使用最大的负次正规数）
    rows_opt[8] = row_fill_bits(BF16_SUB_MAX_NEG)

    # 行 9: 混合正常值（重复模式填满一行）
    mixed_normal = [
        1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 10.0, -10.0,
    ]
    rows_opt[9] = row_from_list(mixed_normal)

    # 行 10: 混合特殊值（包含 +Inf, -Inf, NaN, +0.0, -0.0, 1.0, -1.0）
    # 使用 float 值构造；NaN 会产生一个 canonical NaN 比特型
    mixed_special = [
        float('inf'), float('-inf'), float('nan'), 0.0, -0.0, 1.0, -1.0,
    ]
    rows_opt[10] = row_from_list(mixed_special)

    # 行 11: 交替 +0.0 和 -0.0（按索引奇偶）
    bits = np.zeros(COLS, dtype=np.uint16)
    bits[1::2] = BF16_NEG_ZERO  # 奇数索引为 -0.0
    rows_opt[11] = bf16_from_bits_u16(bits)

    # 行 12: 均匀随机 [-10.0, 10.0]
    r12 = torch.empty(COLS, dtype=torch.float32).uniform_(-10.0, 10.0).to(torch.bfloat16)
    rows_opt[12] = r12

    # 行 13: 均匀随机 [-1000.0, 1000.0]
    r13 = torch.empty(COLS, dtype=torch.float32).uniform_(-1000.0, 1000.0).to(torch.bfloat16)
    rows_opt[13] = r13

    # 行 14: 均匀随机 [-0.1, 0.1]
    r14 = torch.empty(COLS, dtype=torch.float32).uniform_(-0.1, 0.1).to(torch.bfloat16)
    rows_opt[14] = r14

    # 行 15: 均匀随机 [0.0, 1.0]
    r15 = torch.empty(COLS, dtype=torch.float32).uniform_(0.0, 1.0).to(torch.bfloat16)
    rows_opt[15] = r15

    # 行 16: 正态分布 (均值0, 标准差5)
    r16 = (torch.randn(COLS, dtype=torch.float32) * 5.0).to(torch.bfloat16)
    rows_opt[16] = r16

    # 行 17: 随机值注入 Inf/NaN
    base17 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits17 = torch_bf16_to_uint16(base17)
    inj_pos = [0, 17, 31, 63, 127, 255, 511, 767]
    for k, p in enumerate(inj_pos):
        if p < COLS:
            bits17[p] = BF16_INF_POS if (k % 2 == 0) else BF16_QNAN
    rows_opt[17] = bf16_from_bits_u16(bits17)

    # 行 18: 随机值注入次正规数（交替最小正/负次正规数）
    base18 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits18 = torch_bf16_to_uint16(base18)
    for k, p in enumerate(inj_pos):
        if p < COLS:
            bits18[p] = 0x0001 if (k % 2 == 0) else 0x8001
    rows_opt[18] = bf16_from_bits_u16(bits18)

    # 行 19: 随机大正数 [0.0, BF16_MAX/2]
    # 将 BF16_MAX 位型转为 float32，再除以 2 作为上界
    max_f32 = bf16_from_bits_u16(np.array([BF16_MAX], dtype=np.uint16)).float()[0].item()
    high = max_f32 / 2.0
    # 为避免极端溢出，uniform 上下界用 float32 承载
    r19 = torch.empty(COLS, dtype=torch.float32).uniform_(0.0, float(high)).to(torch.bfloat16)
    rows_opt[19] = r19

    # 行 20: 大正值递增序列 [100.0, 100.1, ...]
    r20 = (torch.arange(COLS, dtype=torch.float32) * 0.1 + 100.0).to(torch.bfloat16)
    rows_opt[20] = r20

    # 行 21: 大负值递减序列 [-100.0, -100.1, ...]
    r21 = (-(torch.arange(COLS, dtype=torch.float32) * 0.1 + 100.0)).to(torch.bfloat16)
    rows_opt[21] = r21

    # 行 22: 单个元素远大于其他（中间元素=10.0，其余=0.0）
    mid = COLS // 2
    r22 = torch.zeros(COLS, dtype=torch.bfloat16)
    r22[mid] = torch.tensor(10.0, dtype=torch.bfloat16)
    rows_opt[22] = r22

    # 行 23: 单个 +Inf 元素（中间元素=+Inf，其余=0.0）
    bits23 = np.full(COLS, BF16_POS_ZERO, dtype=np.uint16)
    bits23[mid] = BF16_INF_POS
    rows_opt[23] = bf16_from_bits_u16(bits23)

    # 行 24: 两个 +Inf 元素（位置在 1/3 与 2/3 处，其余=0.0）
    i1, i2 = COLS // 3, (2 * COLS) // 3
    bits24 = np.full(COLS, BF16_POS_ZERO, dtype=np.uint16)
    bits24[i1] = BF16_INF_POS
    bits24[i2] = BF16_INF_POS
    rows_opt[24] = bf16_from_bits_u16(bits24)

    # 行 25: 单个 NaN 元素（中间元素=NaN，其余=0.0）
    bits25 = np.full(COLS, BF16_POS_ZERO, dtype=np.uint16)
    bits25[mid] = BF16_QNAN
    rows_opt[25] = bf16_from_bits_u16(bits25)

    # 行 26: 零均值、非零方差 [1.0, -1.0, ...]
    r26 = torch.ones(COLS, dtype=torch.bfloat16)
    r26[1::2] = torch.tensor(-1.0, dtype=torch.bfloat16)
    rows_opt[26] = r26

    # 行 27: 值接近常数 (小方差) [5.000, 5.001, 5.002, ...]
    r27 = (5.0 + 0.001 * torch.arange(COLS, dtype=torch.float32)).to(torch.bfloat16)
    rows_opt[27] = r27

    # 行 28: 包含 Inf/NaN（在随机数中注入）
    base28 = torch.empty(COLS, dtype=torch.float32).uniform_(-3.0, 3.0).to(torch.bfloat16)
    bits28 = torch_bf16_to_uint16(base28)
    inj_pos = [0, 29, 57, 101, 211, 333, 455, 699]  # 分散的位置
    for k, p in enumerate(inj_pos):
        if p < COLS:
            bits28[p] = BF16_INF_POS if (k % 2 == 0) else BF16_QNAN
    rows_opt[28] = bf16_from_bits_u16(bits28)

    # 行 29: 所有值均相差 epsilon [0, eps, 2*eps, ...]
    r29 = (EPSILON_LN * torch.arange(COLS, dtype=torch.float32)).to(torch.bfloat16)
    rows_opt[29] = r29

    # 行 30: 大数值 [0.0, BF16_MAX/10]
    high10 = float(max_f32 / 10.0)
    r30 = torch.empty(COLS, dtype=torch.float32).uniform_(0.0, high10).to(torch.bfloat16)
    rows_opt[30] = r30

    # 行 31: 小数值 (非次正规) [BF16_MIN_NORMAL, BF16_MIN_NORMAL*10]
    # 先以 float32 生成，再量化为 bf16，确保非次正规范围
    min_norm_f32 = bf16_from_bits_u16(np.array([BF16_MIN_NORMAL], dtype=np.uint16)).float()[0].item()
    r31 = torch.empty(COLS, dtype=torch.float32).uniform_(float(min_norm_f32), float(min_norm_f32 * 10.0)).to(torch.bfloat16)
    rows_opt[31] = r31

    # 行 32: 零均值、非零均方根 [1.0, -1.0, ...]
    r32 = torch.ones(COLS, dtype=torch.bfloat16)
    r32[1::2] = torch.tensor(-1.0, dtype=torch.bfloat16)
    rows_opt[32] = r32

    # 行 33: 值接近零 (小均方根) [0.000, 0.001, 0.002, ...]
    r33 = (0.001 * torch.arange(COLS, dtype=torch.float32)).to(torch.bfloat16)
    rows_opt[33] = r33

    # 行 34: 包含 Inf/NaN（在随机数中注入）
    base34 = torch.empty(COLS, dtype=torch.float32).uniform_(-2.5, 2.5).to(torch.bfloat16)
    bits34 = torch_bf16_to_uint16(base34)
    inj_pos2 = [5, 23, 57, 89, 233, 377, 521, 733]
    for k, p in enumerate(inj_pos2):
        if p < COLS:
            bits34[p] = BF16_INF_POS if (k % 2 == 0) else BF16_QNAN
    rows_opt[34] = bf16_from_bits_u16(bits34)

    # 行 35: 所有值均相差 epsilon (从零开始) [0, eps, 2*eps, ...]
    r35 = (EPSILON_RMS * torch.arange(COLS, dtype=torch.float32)).to(torch.bfloat16)
    rows_opt[35] = r35

    # 行 36: 大数值 [0.0, BF16_MAX/10]
    high10_b = float(max_f32 / 10.0)
    r36 = torch.empty(COLS, dtype=torch.float32).uniform_(0.0, high10_b).to(torch.bfloat16)
    rows_opt[36] = r36

    # 行 37: 小数值 (非次正规) [BF16_MIN_NORMAL, BF16_MIN_NORMAL*10]
    min_norm_f32_b = bf16_from_bits_u16(np.array([BF16_MIN_NORMAL], dtype=np.uint16)).float()[0].item()
    r37 = torch.empty(COLS, dtype=torch.float32).uniform_(float(min_norm_f32_b), float(min_norm_f32_b * 10.0)).to(torch.bfloat16)
    rows_opt[37] = r37

    # 行 38: 零点附近密集采样 [-5, ..., 0, ..., 5]
    r38 = torch.linspace(-5.0, 5.0, COLS, dtype=torch.float32).to(torch.bfloat16)
    rows_opt[38] = r38

    # 行 39: 大正值 [10, ..., 100]
    r39 = torch.linspace(10.0, 100.0, COLS, dtype=torch.float32).to(torch.bfloat16)
    rows_opt[39] = r39

    # 行 40: 大负值 [-10, ..., -100]
    r40 = torch.linspace(-10.0, -100.0, COLS, dtype=torch.float32).to(torch.bfloat16)
    rows_opt[40] = r40

    # 行 41: 包含 +Inf/-Inf/NaN（在随机数中注入）
    base41 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits41 = torch_bf16_to_uint16(base41)
    # 以三元循环的方式注入：+Inf, -Inf, NaN
    inj_pos3 = [7, 49, 97, 149, 223, 307, 401, 503, 607, 701]
    for k, p in enumerate(inj_pos3):
        if p < COLS:
            mode = k % 3
            if mode == 0:
                bits41[p] = BF16_INF_POS
            elif mode == 1:
                bits41[p] = BF16_INF_NEG
            else:
                bits41[p] = BF16_QNAN
    rows_opt[41] = bf16_from_bits_u16(bits41)

    # 行 42: 零点附近密集采样（erf/GELU 检验） [-5, ..., 0, ..., 5]
    r42 = torch.linspace(-5.0, 5.0, COLS, dtype=torch.float32).to(torch.bfloat16)
    rows_opt[42] = r42

    # 行 43: 大正值 [10, ..., 100]（GELU(x) ~ x 区域）
    r43 = torch.linspace(10.0, 100.0, COLS, dtype=torch.float32).to(torch.bfloat16)
    rows_opt[43] = r43

    # 行 44: 大负值 [-10, ..., -100]（GELU(x) ~ 0 区域）
    r44 = torch.linspace(-10.0, -100.0, COLS, dtype=torch.float32).to(torch.bfloat16)
    rows_opt[44] = r44

    # 行 45: 包含 +Inf/-Inf/NaN（在随机数中注入）
    base45 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits45 = torch_bf16_to_uint16(base45)
    inj_pos4 = [3, 37, 83, 131, 197, 269, 347, 419, 587, 659, 731]
    for k, p in enumerate(inj_pos4):
        if p < COLS:
            mode = k % 3
            if mode == 0:
                bits45[p] = BF16_INF_POS
            elif mode == 1:
                bits45[p] = BF16_INF_NEG
            else:
                bits45[p] = BF16_QNAN
    rows_opt[45] = bf16_from_bits_u16(bits45)

    # 行 46: 随机正数 [0.0, 10.0]
    r46 = torch.empty(COLS, dtype=torch.float32).uniform_(0.0, 10.0).to(torch.bfloat16)
    rows_opt[46] = r46

    # 行 47: 随机负数 [-10.0, 0.0]
    r47 = torch.empty(COLS, dtype=torch.float32).uniform_(-10.0, 0.0).to(torch.bfloat16)
    rows_opt[47] = r47

    # 行 48: 大数值 [BF16_MAX/2, BF16_MAX]
    max_low = float(max_f32 / 2.0)
    r48 = torch.empty(COLS, dtype=torch.float32).uniform_(max_low, float(max_f32)).to(torch.bfloat16)
    rows_opt[48] = r48

    # 行 49: 小数值 [BF16_MIN_NORMAL, BF16_MIN_NORMAL*10]
    min_norm_for_49 = bf16_from_bits_u16(np.array([BF16_MIN_NORMAL], dtype=np.uint16)).float()[0].item()
    r49 = torch.empty(COLS, dtype=torch.float32).uniform_(float(min_norm_for_49), float(min_norm_for_49 * 10.0)).to(torch.bfloat16)
    rows_opt[49] = r49

    # 行 50: 包含 +Inf（在随机数中注入）
    base50 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits50 = torch_bf16_to_uint16(base50)
    inj_pos_inf = [11, 79, 143, 211, 307, 419, 523, 647, 701, 757]
    for p in inj_pos_inf:
        if p < COLS:
            bits50[p] = BF16_INF_POS
    rows_opt[50] = bf16_from_bits_u16(bits50)

    # 行 51: 包含 -Inf（在随机数中注入）
    base51 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits51 = torch_bf16_to_uint16(base51)
    inj_pos_ninf = [7, 63, 127, 193, 271, 349, 509, 569, 631, 743]
    for p in inj_pos_ninf:
        if p < COLS:
            bits51[p] = BF16_INF_NEG
    rows_opt[51] = bf16_from_bits_u16(bits51)

    # 行 52: 包含 NaN（在随机数中注入）
    base52 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits52 = torch_bf16_to_uint16(base52)
    inj_pos_nan = [5, 41, 97, 157, 233, 307, 389, 457, 613, 727]
    for p in inj_pos_nan:
        if p < COLS:
            bits52[p] = BF16_QNAN
    rows_opt[52] = bf16_from_bits_u16(bits52)

    # 行 53: 包含 0.0（在随机数中注入 +0.0）
    base53 = torch.empty(COLS, dtype=torch.float32).uniform_(-5.0, 5.0).to(torch.bfloat16)
    bits53 = torch_bf16_to_uint16(base53)
    inj_pos_zero = [3, 29, 73, 109, 181, 269, 347, 463, 557, 739]
    for p in inj_pos_zero:
        if p < COLS:
            bits53[p] = BF16_POS_ZERO
    rows_opt[53] = bf16_from_bits_u16(bits53)

    # 行 54: 对数正态分布（mu=0, sigma=1），正数，适中范围
    gen54 = torch.Generator(device='cpu').manual_seed(SEED + 54)
    z54 = torch.randn(COLS, dtype=torch.float32, generator=gen54)
    r54 = torch.exp(z54).to(torch.bfloat16)
    rows_opt[54] = r54

    # 行 55: 指数分布（lambda=1），使用 -ln(1-U)
    gen55 = torch.Generator(device='cpu').manual_seed(SEED + 55)
    u55 = torch.rand(COLS, dtype=torch.float32, generator=gen55)
    r55 = (-torch.log1p(-u55)).to(torch.bfloat16)
    rows_opt[55] = r55

    # 行 56: 很小的均匀分布（非次正规为主），[-1e-3, 1e-3]
    gen56 = torch.Generator(device='cpu').manual_seed(SEED + 56)
    r56 = torch.empty(COLS, dtype=torch.float32).uniform_(-1e-3, 1e-3, generator=gen56).to(torch.bfloat16)
    rows_opt[56] = r56

    # 行 57: 复用行12的范围但使用不同随机种子，均匀[-10, 10]
    gen57 = torch.Generator(device='cpu').manual_seed(SEED + 57)
    r57 = torch.empty(COLS, dtype=torch.float32).uniform_(-10.0, 10.0, generator=gen57).to(torch.bfloat16)
    rows_opt[57] = r57

    # 行 58: 病态交替极值（+BF16_MAX, -BF16_MAX 交替）
    bits58 = np.empty(COLS, dtype=np.uint16)
    bits58[0::2] = BF16_MAX
    bits58[1::2] = BF16_MAX_NEG
    rows_opt[58] = bf16_from_bits_u16(bits58)

    # 行 59: 锯齿大范围线性 [-1000, 1000]
    r59 = torch.linspace(-1000.0, 1000.0, COLS, dtype=torch.float32).to(torch.bfloat16)
    rows_opt[59] = r59

    # 行 60: 几何级数（2^exp，exp 从 -10 到 10）
    exp60 = torch.linspace(-10.0, 10.0, COLS, dtype=torch.float32)
    r60 = torch.pow(2.0, exp60).to(torch.bfloat16)
    rows_opt[60] = r60

    # 行 61: 交替次正规与带符号零（sub+ / +0 / sub- / -0 重复）
    pattern61 = np.array([0x007F, 0x0000, 0x807F, 0x8000], dtype=np.uint16)
    reps61 = (COLS + pattern61.size - 1) // pattern61.size
    bits61 = np.tile(pattern61, reps61)[:COLS]
    rows_opt[61] = bf16_from_bits_u16(bits61)

    # 行 62: 随机整数并转浮点，范围 [-2048, 2048]
    gen62 = torch.Generator(device='cpu').manual_seed(SEED + 62)
    ints62 = torch.randint(low=-2048, high=2049, size=(COLS,), dtype=torch.int32, generator=gen62)
    r62 = ints62.to(torch.float32).to(torch.bfloat16)
    rows_opt[62] = r62

    # 行 63: 病态交替极值（与行58相同策略，但起始顺序反转为 -MAX, +MAX）
    bits63 = np.empty(COLS, dtype=np.uint16)
    bits63[0::2] = BF16_MAX_NEG  # 偶数索引：-BF16_MAX
    bits63[1::2] = BF16_MAX      # 奇数索引：+BF16_MAX
    rows_opt[63] = bf16_from_bits_u16(bits63)

    # type narrowing for linters
    assert all(row is not None for row in rows_opt)
    rows: List[torch.Tensor] = [row for row in rows_opt if row is not None]
    return rows


def main():
    # Set deterministic seeds
    try:
        torch.manual_seed(SEED)
    except Exception:
        pass
    try:
        np.random.seed(SEED)
    except Exception:
        pass

    print(f"Generating X_test_tensor_bf16.bin with ROWS={ROWS}, COLS={COLS}, SEED={SEED} ...")
    rows = build_rows()
    x_bf16 = torch.cat(rows, dim=0)  # shape: (ROWS*COLS,)

    # Save as uint16 bf16 bit patterns
    u16 = torch_bf16_to_uint16(x_bf16)
    u16.tofile(OUT_FILE)

    print(f"Saved: {OUT_FILE}")
    print(f"Total elements: {u16.size}")
    # Quick stats (avoid NaN/Inf issues by masking finite values)
    f32 = x_bf16.float()
    finite_mask = torch.isfinite(f32)
    if finite_mask.any():
        f32_finite = f32[finite_mask]
        print(f"Finite stats: min={float(f32_finite.min()):.6e}, max={float(f32_finite.max()):.6e}")
    # Show a few head values (as bf16 and bits)
    head = x_bf16[:16]
    head_bits = torch_bf16_to_uint16(head)
    print("Head bf16 values (float32 view):", head.float().tolist())
    print("Head bf16 bits (hex):", [f"0x{int(b):04x}" for b in head_bits[:16]])


if __name__ == '__main__':
    main()
