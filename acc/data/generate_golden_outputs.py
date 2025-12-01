#!/usr/bin/env python3
"""
Generate golden output data for all configurations
"""

import numpy as np
import torch
import struct
import os
import time
import json

# Module-level data directory constant: directory of this script
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

# === Hardware-aligned FP32 -> BF16 conversion (matches current fp32_to_bf16.v) ===
def float32_to_bf16_uint16_hw(x):

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    u = x.view(np.uint32)

    sign = (u >> 31) & 0x1
    exp  = (u >> 23) & 0xFF
    frac = u & 0x7FFFFF

    is_nan = (exp == 0xFF) & (frac != 0)
    is_inf = (exp == 0xFF) & (frac == 0)
    is_sub_or_zero = (exp == 0x00)

    out = np.zeros_like(u, dtype=np.uint32)

    # NaN: sNaN -> qNaN; ensure payload non-zero in bf16
    if np.any(is_nan):
        frac_hi7 = ((frac >> 16) & 0x7F).copy()
        # ensure non-zero payload
        frac_hi7 = np.where(frac_hi7 == 0, 0x40, frac_hi7)
        # sNaN if frac[22]==0 -> set quiet bit
        is_snan = ((frac >> 22) & 0x1) == 0
        frac_hi7 = np.where(is_nan & is_snan, frac_hi7 | 0x40, frac_hi7)
        out_nan = (sign << 15) | (0xFF << 7) | frac_hi7
        out = np.where(is_nan, out_nan, out)

    # Inf
    if np.any(is_inf):
        out_inf = (sign << 15) | (0xFF << 7)
        out = np.where(is_inf, out_inf, out)

    # Zero or subnormal -> FTZ to signed zero
    if np.any(is_sub_or_zero):
        out_zer = (sign << 15)
        out = np.where(is_sub_or_zero, out_zer, out)

    # Normal numbers -> RNE (ties-to-even)
    is_norm = ~(is_nan | is_inf | is_sub_or_zero)
    if np.any(is_norm):
        u_norm = u.copy()
        bias = (0x7FFF + ((u_norm >> 16) & 0x1)).astype(np.uint32)
        v = (u_norm + bias).astype(np.uint32)
        out_norm = (v >> 16) & 0xFFFF
        out = np.where(is_norm, out_norm, out)

    return out.astype(np.uint16)

def uint16_to_torch_bf16(data_uint16):
    """Convert uint16 bit pattern to torch.bfloat16"""
    # Parse bf16 format by bits: 1 sign bit + 8 exponent bits + 7 mantissa bits
    bf16_values = []
    for val in data_uint16:
        # Extract parts of bf16
        sign = (val >> 15) & 0x1
        exp = (val >> 7) & 0xFF
        mantissa = val & 0x7F
        
        # Special value handling
        if exp == 0:
            # Zero or subnormal number
            if mantissa == 0:
                # Preserve signed zero per IEEE 754
                result = -0.0 if sign == 1 else 0.0
            else:
                # Subnormal number (rare in bf16)
                result = (1 if sign == 0 else -1) * (mantissa / 128.0) * (2 ** -126)
        elif exp == 255:
            # Infinity or NaN
            result = float('inf') if mantissa == 0 else float('nan')
            if sign == 1:
                result = -result
        else:
            # Normal number
            # bf16 bias is 127 (same as float32)
            exp_val = exp - 127.0
            mantissa_val = 1.0 + (mantissa / 128.0)
            result = (1.0 if sign == 0 else -1.0) * mantissa_val * (2.0 ** exp_val)
        
        # Directly use bf16 for calculation to avoid float32 precision loss
        bf16_val = torch.tensor(result, dtype=torch.bfloat16)
        bf16_values.append(bf16_val)
    
    return torch.stack(bf16_values)

def torch_bf16_to_uint16(tensor):
    """Convert torch.bfloat16 to uint16 bit pattern"""
    # Convert to float32, then get its bit pattern
    float32_values = tensor.float()
    uint32_values = np.array([struct.unpack('I', struct.pack('f', float(val)))[0] for val in float32_values], dtype=np.uint32)
    # Extract bit pattern of bfloat16 (sign bit + 8 exponent bits + 7 mantissa bits)
    # The trick here is to right shift 16 bits, discarding the lower 16 bits of float32 (i.e., the lower 16 bits of the mantissa)
    uint16_values = (uint32_values >> 16).astype(np.uint16)
    return uint16_values
def eltwise_add(x1, x2):
    """Element-wise addition"""
    result = x1 + x2
    
    # Output debug information only at error points
    # Use hardware-aligned conversion on float32 view
    python_result_uint16 = float32_to_bf16_uint16_hw(result.float())
    
    # Check for errors (compare with previously generated golden data)
    try:
        # Updated to compare against file named by function
        golden_uint16 = np.fromfile(DATA_DIR + 'eltwise_add_bf16.bin', dtype=np.uint16)
        error_count = 0
        for i in range(min(100, len(python_result_uint16))):
            if python_result_uint16[i] != golden_uint16[i]:
                error_count += 1
                if error_count <= 10:  # Only show the first 10 errors
                    print(f"Python Error[{i}]: expected={golden_uint16[i]}, actual={int(python_result_uint16[i])}")
                    a_u16 = float32_to_bf16_uint16_hw(x1[i:i+1].float())[0]
                    b_u16 = float32_to_bf16_uint16_hw(x2[i:i+1].float())[0]
                    print(
                        f"    bf16add debug: a={int(a_u16)} (0x{int(a_u16):04x}), "
                        f"b={int(b_u16)} (0x{int(b_u16):04x}) -> result={int(python_result_uint16[i])} "
                        f"(0x{int(python_result_uint16[i]):04x})"
                    )
        if error_count > 0:
            print(f"Python total errors: {error_count} (in first 100 points)")
    except:
        pass  # If the golden file does not exist, do not output debug information
    
    return result

def safe_softmax(x, axis=-1):
    """Safe softmax (along given axis)."""
    max_ = torch.max(x, dim=axis, keepdim=True)[0]
    sub_ = x - max_
    exp_ = torch.exp(sub_)
    sum_ = torch.sum(exp_, dim=axis, keepdim=True)
    return exp_ / sum_

def eltwise_mul(x1, x2):
    """Element-wise multiplication (bf16 aware via torch)."""
    result = x1 * x2
    
    # Output debug information only at error points
    python_result_uint16 = float32_to_bf16_uint16_hw(result.float())
    return result

def sigmoid(x):
    """Sigmoid activation function with input clamp to [-500, 500]."""
    x = torch.clamp(x, -500, 500)
    return 1 / (1 + torch.exp(-x))

def silu(x):
    """SiLU activation function"""
    return x * sigmoid(x)

def rms_norm(x, weight=None, eps=1e-6):
    """RMS normalization"""
    pow_ = x**2
    mean_ = torch.mean(pow_, dim=-1, keepdim=True)
    rms_ = torch.sqrt(mean_ + eps)
    x_norm = x / rms_
    if weight is not None:
        x_norm = x_norm * weight
    return x_norm

def layer_norm(x, weight=None, bias=None, eps=1e-6):
    """Layer normalization"""
    mean_ = torch.mean(x, dim=-1, keepdim=True)
    sub_ = x - mean_
    pow_ = torch.pow(sub_, 2)
    var_ = torch.mean(pow_, dim=-1, keepdim=True)
    sqrt_ = torch.sqrt(var_ + eps)
    norm_ = sub_ / sqrt_
    if weight is not None:
        norm_ = norm_ * weight
    if bias is not None:
        norm_ = norm_ + bias
    return norm_

def sigmoid_bf16(x_bf16):
    """Sigmoid on bf16 input, with clamp to match C++ implementation."""
    x_bf16 = x_bf16.to(torch.bfloat16)
    x_fp32 = x_bf16.float()
    x_fp32 = torch.clamp(x_fp32, -500, 500)
    y_fp32 = 1 / (1 + torch.exp(-x_fp32))
    return y_fp32.to(torch.bfloat16)

def gelu_bf16(x_bf16, approximate: str = 'tanh'):
    """GELU on bf16 input. approximate can be 'none' or 'tanh' (PyTorch semantics)."""
    x_bf16 = x_bf16.to(torch.bfloat16)
    x_fp32 = x_bf16.float()
    y_fp32 = torch.nn.functional.gelu(x_fp32, approximate=approximate)
    return y_fp32.to(torch.bfloat16)

def generate_golden_outputs():
    """Generate golden outputs for all configurations"""
    print("Generating golden output data...")
    
    # Use module-level DATA_DIR
    
    # Load input data
    print("Loading input data...")
    in0_uint16 = np.fromfile(DATA_DIR + 'X_test_tensor_bf16.bin', dtype=np.uint16)
    in1_uint16 = np.fromfile(DATA_DIR + 'Y_test_tensor_bf16.bin', dtype=np.uint16)
    
    # Convert to torch.bfloat16
    in0 = uint16_to_torch_bf16(in0_uint16)
    in1 = uint16_to_torch_bf16(in1_uint16)
    
    # Shape/length reconciliation for element-wise ops
    def describe_shape(vec: torch.Tensor):
        n = vec.numel()
        rows = n // 768 if n % 768 == 0 else None
        return n, rows

    n0, r0 = describe_shape(in0)
    n1, r1 = describe_shape(in1)
    print(f"X length: {n0} (rows={r0 if r0 is not None else 'N/A'})")
    print(f"Y length: {n1} (rows={r1 if r1 is not None else 'N/A'})")

    if n0 != n1:
        # Try to align Y (in1) to X (in0) by row tiling or broadcasting
        if r0 is not None:
            if r1 is not None and r0 % r1 == 0:
                # Tile Y rows to match X rows
                factor = r0 // r1
                in1 = in1.float().reshape(r1, 768)
                in1 = in1.repeat_interleave(factor, dim=0).to(torch.bfloat16).reshape(-1)
                n1 = in1.numel(); r1 = r0
                print(f"Aligned Y by tiling rows x{factor} -> length={n1}, rows={r1}")
            elif n1 == 768:
                # Broadcast single row of Y to all X rows
                in1 = in1.float().reshape(1, 768).repeat(r0, 1).to(torch.bfloat16).reshape(-1)
                n1 = in1.numel(); r1 = r0
                print(f"Broadcasted Y single row to X rows -> length={n1}, rows={r1}")
            else:
                raise ValueError(f"Shape mismatch: X has {n0} (rows={r0}), Y has {n1} (rows={r1}). Cannot align.")
        else:
            if n1 == n0:
                pass
            else:
                raise ValueError(f"Shape mismatch: X length {n0}, Y length {n1}.")
    
    print(f"Using data size: {len(in0)} elements")
    
    # Generate golden outputs for all configurations
    configs = list(range(7))
    config_to_name = {
        0: 'eltwise_add',
        1: 'safe_softmax',
        2: 'eltwise_mul',
        3: 'gelu_bf16',
        4: 'silu',
        5: 'rms_norm',
        6: 'layer_norm',
    }
    # 收集 CPU 端各算子的延时（纳秒）
    cpu_latency_results = []
    
    for config in configs:
        # If the data is a multiple of 768, treat it as [rows, 768] for row-wise ops
        rows = None
        if len(in0) % 768 == 0:
            rows = len(in0) // 768
        op_name = config_to_name.get(config, f'config_{config}')
        # 仅对“计算阶段”计时（不包含后续展平/补零/写文件）
        start_ns = time.perf_counter_ns()
        if config == 0:
            # Element-wise addition: X + Y
            output_data_bf16 = eltwise_add(in0, in1)
        elif config == 1:
            # Safe softmax (row-wise if applicable)
            if rows is not None:
                x2d = in0.float().reshape(rows, 768)
                y2d = safe_softmax(x2d, axis=-1)
                output_data_bf16 = y2d.to(torch.bfloat16).reshape(-1)
            else:
                output_data_bf16 = safe_softmax(in0.float()).to(torch.bfloat16)
        elif config == 2:
            # Element-wise multiplication: X * Y
            output_data_bf16 = eltwise_mul(in0, in1)
        elif config == 3:
            # GELU (X only)
            output_data_bf16 = gelu_bf16(in0)
        elif config == 4:
            # SiLU
            output_data_bf16 = silu(in0.float()).to(torch.bfloat16)
        elif config == 5:
            # RMS normalization (row-wise if applicable)
            if rows is not None:
                x2d = in0.float().reshape(rows, 768)
                y2d = rms_norm(x2d)
                output_data_bf16 = y2d.to(torch.bfloat16).reshape(-1)
            else:
                output_data_bf16 = rms_norm(in0.float()).to(torch.bfloat16)
        elif config == 6:
            # Layer normalization (row-wise if applicable)
            if rows is not None:
                x2d = in0.float().reshape(rows, 768)
                y2d = layer_norm(x2d)
                output_data_bf16 = y2d.to(torch.bfloat16).reshape(-1)
            else:
                output_data_bf16 = layer_norm(in0.float()).to(torch.bfloat16)
        else:
            output_data_bf16 = torch.zeros_like(in0)
        end_ns = time.perf_counter_ns()
        cpu_ns = int(end_ns - start_ns)

        # 记录并打印 CPU 延时（纳秒/毫秒）
        cpu_latency_results.append({
            "op": op_name,
            "config": int(config),
            "elements": int(len(in0)),
            "rows": int(rows) if rows is not None else None,
            "latency_ns": cpu_ns,
            "latency_ms": cpu_ns / 1e6,
        })
        print(f"  [CPU] {op_name}: latency = {cpu_ns} ns ({cpu_ns/1e6:.3f} ms)")
        
        # Ensure the result is 1D
        if output_data_bf16.dim() > 1:
            output_data_bf16 = output_data_bf16.flatten()
        
        # Truncate or pad to the correct length
        if len(output_data_bf16) > len(in0):
            output_data_bf16 = output_data_bf16[:len(in0)]
        elif len(output_data_bf16) < len(in0):
            padding = torch.zeros(len(in0) - len(output_data_bf16), dtype=torch.bfloat16)
            output_data_bf16 = torch.cat([output_data_bf16, padding])
        
        # Convert to uint16 (hardware-aligned) and save
        # Use float32 view to feed hardware-like converter (RNE + FTZ + NaN rules)
        result_uint16 = float32_to_bf16_uint16_hw(output_data_bf16.float())
        output_file = DATA_DIR + f'{op_name}_bf16.bin'
        result_uint16.tofile(output_file)

        print(f"  Saved to: {output_file}")
        print(f"  Result stats: min={float(output_data_bf16.min()):.6f}, max={float(output_data_bf16.max()):.6f}")
    
    # 写入 CPU 延时统计到 JSON 文件
    try:
        latency_path = os.path.join(DATA_DIR, 'cpu_latency_results.json')
        with open(latency_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_ops": len(cpu_latency_results),
                    "elements": int(len(in0)),
                    "rows": int(len(in0)//768) if len(in0) % 768 == 0 else None,
                },
                "results": cpu_latency_results
            }, f, ensure_ascii=False, indent=2)
        print(f"CPU latency results saved to: {latency_path}")
    except Exception as e:
        print(f"[WARN] Failed to save CPU latency results: {e}")

    print("Golden output generation complete!")

if __name__ == "__main__":
    generate_golden_outputs()
