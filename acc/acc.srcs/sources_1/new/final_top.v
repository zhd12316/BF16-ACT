`timescale 1ns / 1ps

// ==========================================
// 功能：支持7种神经网络操作的硬件加速
// 数据流：BF16输入 -> FP32计算 -> BF16输出
// ==========================================
module final_top(
    input  wire        clk,
    input  wire        rst,
    input  wire [511:0]  data_in,        // BF16输入：32个16位数（512bit）
    input  wire        data_valid,       // 输入有效脉冲
    input  wire        start,            // 启动信号
    input  wire  [2:0]  mode,            // 操作模式
    output reg [511:0] out_data,         // BF16输出：32个16位数
    output reg [1023:0]out_data_el,      
    output reg         out_valid,        // 输出有效信号
    output reg         done,             // 单周期完成脉冲
    output wire [31:0] count             // start到done的周期数
);
    // =================================
    // 操作模式定义（3位编码）
    // =================================
    parameter softmax_mode = 3'b000 ,    // Softmax归一化
              layernorm_mode = 3'b001 ,  // 层归一化
              RMSnorm_mode  = 3'b010 ,   // 均方根归一化
              silu_mode    = 3'b011 ,    // SiLU激活函数
              el_add_mode    = 3'b100 ,  // 逐元素加法
              el_mul_mode  = 3'b101,     // 逐元素乘法
              gelu_mode    = 3'b110 ;    // GELU激活函数

    // =================================
    // 核心数据格式转换信号
    // =================================
    wire [1023:0] fp32_data_in;          // BF16转FP32：32×16bit -> 32×32bit
    wire          fp32_valid_in;         
    reg  [1023:0] fp32_data;             

    // =================================
    // 算术运算单元信号组
    // =================================
    
    // 加法器0（支持向量+向量、向量+常数两种模式）
    reg [1023:0] add_in1, add_in2;       // 向量输入
    reg [31:0]   add_in3;                // 标量输入
    reg          add_valid_in;           
    wire [1023:0] add_out;               
    wire         add_valid;              
    reg  [1:0]   mode_add;               // 模式控制：01=向量+向量, 10=向量+常数

    // 加法器1（结构同加法器0）
    reg [1023:0]  add_in1_1, add_in2_1;
    reg [31:0]    add_in3_1;
    reg           add_valid_in_1;
    wire [1023:0] add_out_1;
    wire          add_valid_1;
    reg  [1:0]    mode_add_1;

    // 减法器（仅支持向量-标量模式）
    reg [31:0]   sub_in;                 // 标量输入：要减去的数（如最大值、均值）
    reg          sub_valid_in;
    wire [1023:0] sub_out;               // 32个结果：每个元素都减去sub_in
    wire         sub_valid;

    // 乘法器0（支持向量×向量、向量×常数）
    reg [1023:0]  mul_in1, mul_in2;      // 向量输入
    reg [31:0]    mul_in3;               // 标量输入
    reg           mul_valid_in;
    wire [1023:0] mul_out;
    wire          mul_valid;
    reg  [1:0]    mode_mul;              // 模式：01=向量×向量, 10=向量×常数

    // 乘法器1、2（结构同乘法器0）
    reg [1023:0]  mul_in1_1, mul_in2_1;
    reg [31:0]    mul_in3_1;
    reg           mul_valid_in_1;
    wire [1023:0] mul_out_1;
    wire          mul_valid_1;
    reg  [1:0]    mode_mul_1;

    reg [1023:0]  mul_in1_2, mul_in2_2;
    reg [31:0]    mul_in3_2;
    reg           mul_valid_in_2;
    wire [1023:0] mul_out_2;
    wire          mul_valid_2;
    reg  [1:0]    mode_mul_2;

    // 指数运算单元（用于Softmax的exp(x)和GELU的tanh近似）
    reg [1023:0]  exp_in;                // 32个FP32输入
    reg           exp_valid_in;
    wire [1023:0] exp_out;               // 32个exp结果
    wire          exp_valid;

    // 累加器0（将32个FP32求和为1个FP32标量）
    reg [1023:0]  sum_in;                // 32个FP32输入
    reg           sum_valid_in;
    wire [31:0]   sum_out;               // 标量和
    wire          sum_valid;

    // 累加器1（结构同累加器0）
    reg [1023:0]  sum_in_1;
    reg           sum_valid_in_1;
    wire [31:0]   sum_out_1;
    wire          sum_valid_1;

    // 除法器（支持向量÷向量、向量÷常数）
    reg [1023:0]  div_in1;               // 被除数向量（32个FP32）
    reg [31:0]    div_in2;               // 除数标量
    reg [1023:0]  div_in3;               // 除数向量
    reg           div_valid_in;
    wire [1023:0] div_out;               // 32个商
    wire          div_valid;
    reg  [1:0]    mode_div;              // 模式：01=向量÷向量, 10=向量÷常数

    // 平方根单元（用于归一化中的标准差计算）
    reg  [31:0]   sqrt_in;               // 标量输入
    reg           sqrt_valid_in;
    wire [31:0]   sqrt_out;              // 标量输出
    wire          sqrt_valid;

    // FP32到BF16转换器0（主输出通道）
    reg [1023:0]  fp32_2_bf16_in;        // 32个FP32输入
    reg           fp32_2_bf16_valid;
    wire [511:0]  out_w;                 // 32个BF16输出
    wire          out_valid_w;

    // FP32到BF16转换器1（用于逐元素操作的第二路输出）
    reg [1023:0]  fp32_2_bf16_in_1;
    reg          fp32_2_bf16_valid_1;
    wire [511:0]  out_w_1;
    wire          out_valid_w_1;

    // =================================
    // BRAM控制信号组
    // =================================
    
    // BRAM1：双端口RAM（1536×1024bit）
    // 用途：存储中间FP32数据，如归一化中的原始数据、Softmax的输入数据
    // 端口A：写入，端口B：读取
    reg         wea1, ena1, enb1;        
    reg  [10:0] addra1, addrb1;          // 11位地址：0-1535
    reg  [1023:0] dina1;                 // 写数据：32个FP32
    wire [1023:0] doutb1;                // 读数据：32个FP32

    // BRAM2：双端口RAM（1536×1024bit）
    // 用途：存储exp计算结果、归一化的中间结果等
    reg         wea2, ena2, enb2;
    reg  [10:0] addra2, addrb2;
    reg  [1023:0] dina2;
    wire [1023:0] doutb2;

    // BRAM3：非对称宽度双端口RAM
    // 写端口：3072×512bit（存储BF16数据）
    // 读端口：768×2048bit（一次读出4组数据，用于逐元素操作）
    // 读宽度是写宽度的4倍
    reg         wea3, ena3, enb3;
    reg  [11:0] addra3;                  
    reg  [9:0]  addrb3;                  
    reg  [511:0]  dina3;                 // 写数据：32个BF16
    wire [2047:0] doutb3;                // 读数据：128个BF16（4×32）

    // =================================
    // Softmax专用控制信号
    // 计算流程：
    // 找最大值max(x),计算exp(x - max)避免数值溢出,求和sum(exp(x - max))
    // 归一化：exp(x - max) / sum
    // =================================
    reg [511:0]   softmax_data_in;       
    reg           softmax_data_valid;
    wire          max_valid_softmax;     
    wire [31:0]   max_val_fp32_softmax;  // 最大值（FP32标量）
    wire [1023:0] fp32_data_softmax;     
    wire          div_valid_in_softmax;  
    reg [31:0]    sum_out_softmax;       
    reg [31:0]    max_val_fp32;          
    reg           max_valid;             
    reg           sub_valid_in_final;    // 减法输入有效（延迟一拍）
    reg           sum_valid_softmax;     

    // =================================
    // LayerNorm专用控制信号
    // 计算流程：
    // 计算均值μ = mean(x)
    // 计算方差σ² = mean((x-μ)²)
    // 归一化y = (x-μ) / √(σ²+ε)
    // =================================
    reg  [31:0]   sum_out_layernorm;     // 均值
    reg           sum_out_valid_layernorm;
    wire [31:0]   add_out_layernorm;     // 方差计算的加法结果（σ²+ε）
    wire          add_out_valid_layernorm;
    reg  [31:0]   sqrt_out_layernorm;    // 标准差σ
    reg           sub_valid_in_layernorm; // 减均值输入有效
    reg           sqrt_valid_layernorm;   // 平方根输入有效
    reg           div_valid_in_layernorm; // 归一化除法输入有效
    reg  [31:0]   sum_out1_final, sum_out1_final1; 
    reg           sum_out_valid_final;
    reg  [31:0]   sqrt_out_final;        // 标准差最终值
    reg  [31:0]   sum_out_final;
    reg [1023:0]  add_out_final;
    reg           mul_valid_in_1_final, mul_valid_in_1_final1; 

    // =================================
    // RMSNorm专用控制信号
    // 计算流程：
    // 计算均方mean(x²)
    // 计算均方根RMS = √(mean(x²))
    // 归一化y = x / RMS
    // =================================
    reg  [31:0]   sum_out_rmsnorm;       // 平方和
    reg           sum_valid_rmsnorm;
    wire [31:0]   add_out_rmsnorm;       // 均方根计算中间值
    wire          add_out_valid_rmsnorm;
    reg  [31:0]   sqrt_out_rmsnorm;      // 均方根RMS
    reg           sqrt_valid_rmsnorm;
    reg           div_valid_in_final;    // 除法输入有效

    // =================================
    // SiLU专用控制信号
    // 公式：SiLU(x) = x / (1 + exp(-x))
    // =================================
    reg [1023:0]  fp32_data_in_silu;     // SiLU输入数据
    reg           fp32_valid_in_silu;
    wire [1023:0] inv_out;               // 取反结果（-x）
    wire          inv_valid;
    reg [1023:0]  add_out_silu;          // 1 + exp(-x)

    // =================================
    // 逐元素操作专用信号
    // 特点：BRAM3以4:1宽度比存储数据
    // 一次读出4组512bit，分别转FP32后进行运算
    // =================================
    
    // 四路并行数据流（每路32个BF16）
    reg [511:0]   data_in_1, data_in_2, data_in_3, data_in_4;
    reg           data_valid_1, data_valid_2, data_valid_3, data_valid_4;
    reg           data_valid_1_final, data_valid_2_final;
    reg           data_valid_3_final, data_valid_4_final;
    
    // 四路FP32转换后的数据
    wire[1023:0]  fp32_data_in_1, fp32_data_in_2;
    wire[1023:0]  fp32_data_in_3, fp32_data_in_4;
    wire          fp32_valid_in_1, fp32_valid_in_2;
    wire          fp32_valid_in_3, fp32_valid_in_4;

    reg           start_el_add, start_el_mul; // 逐元素操作启动信号
    reg  [1:0]    el_add_cnt, el_mul_cnt;     // 计数器
    reg           el_cnt_flag;
    reg           add_valid1, mul_valid1;
    reg  [4095:0] dina3_el_add, dina3_el_mul; // 缓存（未使用）

    // =================================
    // GELU专用控制信号
    // 近似公式：GELU(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
    // =================================
    reg  [1023:0] fp32_data_in_gelu;     // GELU输入
    reg           fp32_valid_in_gelu;
    wire [1023:0] mul_in1_gelu, mul_in2_gelu; // 乘法器输入（来自gelu子模块）
    wire          mul_valid_in_gelu;
    reg  [1023:0] mul_out_gelu;          // 乘法结果
    reg           mul_valid_gelu;
    reg  [1023:0] mul_out_final;         // 乘法结果寄存
    reg           mul_valid_final;

    // GELU多级流水线寄存器（用于时序对齐）
    reg  [1023:0] mul_out_2_0, mul_out_2_1, mul_out_2_2, mul_out_2_3;
    reg           mul_valid_2_0, mul_valid_2_1, mul_valid_2_2, mul_valid_2_3;
    
    reg [1023:0]  exp_out_gelu;          // exp结果（用于tanh近似）
    reg           exp_valid_gelu;
    wire [1023:0] exp_out_final_mul;     // exp输出用于乘法
    wire [1023:0] exp_out_final_add;     // exp输出用于加法
    wire          exp_valid_final;
    reg [1023:0]  div_in1_gelu;          // 最终除法输入
    reg           div_valid_in_gelu;

    reg           done_flag;             // 完成标志寄存器

    // =================================
    // 主控制状态机
    // 功能：根据mode信号路由数据流和控制各运算单元
    // =================================
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            // 复位：清零所有控制信号
            out_valid          <= 1'b0;
            add_valid_in       <= 1'b0;
            sub_valid_in       <= 1'b0;
            mul_valid_in       <= 1'b0;
            mul_valid_in_1     <= 1'b0;
            exp_valid_in       <= 1'b0;
            sum_valid_in       <= 1'b0;
            sum_valid_in_1     <= 1'b0;
            div_valid_in       <= 1'b0;
            fp32_2_bf16_valid  <= 1'b0;
            fp32_2_bf16_valid_1<= 1'b0;
            add_valid1         <= 1'b0;
            el_add_cnt         <= 2'b0;
            el_mul_cnt         <= 2'b0;
            mode_add           <= 2'b00;
            mode_add_1         <= 2'b00;
            mode_div           <= 2'b00;
            mode_mul           <= 2'b00;
            mode_mul_1         <= 2'b00;
            mode_mul_2         <= 2'b00;
            wea1               <= 1'b0;
            ena1               <= 1'b0;
            enb1               <= 1'b0;
            wea2               <= 1'b0;
            ena2               <= 1'b0;
            enb2               <= 1'b0;
            wea3               <= 1'b0;
            ena3               <= 1'b0;
            enb3               <= 1'b0;
            el_cnt_flag        <= 1'b0;
        end else begin
            case(mode)
                softmax_mode: begin
                    // ===== Softmax数据流 =====
                    // 输入：data_in(BF16) -> BF16toFP32 -> BRAM1存储
                    softmax_data_in       <= data_in;
                    softmax_data_valid    <= data_valid;
                    
                    // 写入BRAM1并查找最大值
                    wea1                  <= fp32_valid_in;
                    ena1                  <= fp32_valid_in;
                    dina1                 <= fp32_data_in;
                    max_valid             <= max_valid_softmax;
                    
                    // 读BRAM1，减最大值（数值稳定）
                    enb1                  <= max_valid_softmax || ((addrb1 != 11'd1535) && addrb1) || max_valid;
                    max_val_fp32          <= max_val_fp32_softmax;
                    sub_valid_in_final    <= enb1;
                    fp32_data             <= doutb1;
                    sub_in                <= max_val_fp32;
                    sub_valid_in          <= sub_valid_in_final;
                    
                    // 计算exp(x - max)
                    exp_in                <= sub_out;
                    exp_valid_in          <= sub_valid;
                    
                    // 累加所有exp值
                    sum_in                <= exp_out;
                    sum_valid_in          <= exp_valid;
                    sum_valid_softmax     <= sum_valid;
                    sum_out_softmax       <= sum_out;
                    sum_out_final         <= sum_out_softmax;
                    
                    // exp结果写入BRAM2
                    wea2                  <= exp_valid;
                    ena2                  <= exp_valid;
                    dina2                 <= exp_out;
                    enb2                  <= sum_valid || ((addrb2 != 11'd1535) && addrb2) || sum_valid_softmax;
                    
                    // 从BRAM2读exp，除以sum完成归一化
                    div_in1               <= doutb2;
                    div_in2               <= sum_out_final;
                    div_valid_in          <= div_valid_in_softmax;
                    mode_div              <= 2'b10; // 向量÷常数模式
                    
                    // 转回BF16输出
                    fp32_2_bf16_in        <= div_out;
                    fp32_2_bf16_valid     <= div_valid;
                    out_data              <= out_w;
                    out_valid             <= out_valid_w;
                end
                
                layernorm_mode: begin
                    // ===== LayerNorm数据流 =====
                    
                    // BRAM1存原始数据，BRAM2存(x-μ)
                    wea1                  <= fp32_valid_in;
                    ena1                  <= fp32_valid_in;
                    dina1                 <= fp32_data_in; 
                    wea2                  <= sub_valid;
                    ena2                  <= sub_valid;
                    dina2                 <= sub_out;
                    
                    // BRAM读控制
                    enb1                  <= sum_valid_1 || ((addrb1 !== 11'd1535) && addrb1) || sum_out_valid_final;
                    enb2                  <= sqrt_valid_layernorm || ((addrb2 !== 11'd1535) && addrb2) || sqrt_valid;
                    
                    // 计算均值μ = sum(x) / N
                    // 这里N=1536，使用乘以1/N代替除法
                    sum_in                <= mul_out_1;      // sum(x * 1/N)
                    sum_valid_in          <= mul_valid_1;
                    sum_out_layernorm     <= sum_out;        // 均值μ
                    sum_out_valid_layernorm <= sum_valid;

                    // 计算方差σ² = sum((x-μ)²) / N
                    sum_valid_in_1        <= mul_valid_2;
                    sum_in_1              <= mul_out_2;      // sum((x-μ)² * 1/N)
                    sum_out_valid_final   <= sum_valid_1; 
                    sum_out1_final        <= sum_out_1;      // 方差σ²
                    sum_out1_final1       <= sum_out1_final; // 流水线延迟
                    
                    // 计算x - μ
                    fp32_data             <= doutb1;
                    sub_valid_in_layernorm<= enb1;
                    sub_in                <= sum_out1_final1; // 减均值
                    sub_valid_in          <= sub_valid_in_layernorm;
                    
                    // 乘法器0：(x-μ) * (1/N)，用于方差计算
                    mode_mul              <= 2'b10;
                    mul_in1               <= sub_out;
                    mul_in3               <= 32'h3d13cd3a;    // 1/1536的FP32表示
                    mul_valid_in          <= sub_valid;
                    
                    // 乘法器1：(x-μ)²，用于方差计算
                    mode_mul_1            <= 2'b01;
                    mul_in1_1             <= mul_out;
                    mul_in2_1             <= mul_out;
                    mul_valid_in_1        <= mul_valid;

                    // 乘法器2：x * (1/N)，用于均值计算
                    mode_mul_2            <= 2'b10;
                    mul_in1_2             <= fp32_data_in;
                    mul_in3_2             <= 32'h3aaaaaab;    // 另一种1/1536表示
                    mul_valid_in_2        <= fp32_valid_in;
                    
                    // 计算标准差σ = √(σ² + ε)
                    // add_out_layernorm = σ² + ε（ε防止除零）
                    sqrt_in               <= add_out_layernorm;
                    sqrt_valid_in         <= add_out_valid_layernorm;
                    sqrt_valid_layernorm  <= sqrt_valid;
                    sqrt_out_layernorm    <= sqrt_out;
                    sqrt_out_final        <= sqrt_out_layernorm;
                    
                    // 归一化(x-μ) / σ
                    div_valid_in_layernorm<= enb2;
                    div_in1               <= doutb2;          // (x-μ)从BRAM2读取
                    div_in2               <= sqrt_out_final;  // 除以σ
                    div_valid_in          <= div_valid_in_layernorm;
                    mode_div              <= 2'b10;           // 向量÷常数
                    
                    // 转回BF16输出
                    fp32_2_bf16_in        <= div_out;
                    fp32_2_bf16_valid     <= div_valid;
                    out_data              <= out_w;
                    out_valid             <= out_valid_w;
                end
                
                RMSnorm_mode: begin
                    // ===== RMSNorm数据流 =====
                    // 公式：y = x / √(mean(x²))
                    
                    // BRAM1存储原始数据
                    wea1                  <= fp32_valid_in;
                    ena1                  <= fp32_valid_in;
                    dina1                 <= fp32_data_in;
                    enb1                  <= sqrt_valid || ((addrb1 != 11'd1535) && addrb1) || sqrt_valid_rmsnorm;
                    
                    // 计算x²
                    mode_mul              <= 2'b01;           // 向量×向量
                    mul_in1               <= mul_out_1;
                    mul_in2               <= mul_out_1;
                    mul_in3               <= 32'b0;
                    mul_valid_in          <= mul_valid_1;
                    
                    // 计算x * (1/N)
                    mode_mul_1            <= 2'b10;
                    mul_in1_1             <= fp32_data_in;
                    mul_in3_1             <= 32'h3d13cd3a;    // 1/1536
                    mul_valid_in_1        <= fp32_valid_in;
                    
                    // 求和：sum(x²)
                    sum_in                <= mul_out;
                    sum_valid_in          <= mul_valid;
                    sum_out_rmsnorm       <= sum_out;
                    sum_valid_rmsnorm     <= sum_valid;
                    
                    // 计算RMS：√(mean(x²))
                    sqrt_in               <= add_out_rmsnorm; // mean(x²)
                    sqrt_valid_in         <= add_out_valid_rmsnorm;
                    sqrt_out_rmsnorm      <= sqrt_out;
                    sqrt_out_final        <= sqrt_out_rmsnorm;
                    sqrt_valid_rmsnorm    <= sqrt_valid;
                    div_valid_in_final    <= enb1;
                    
                    // 归一化：x / RMS
                    div_in1               <= doutb1;
                    div_in2               <= sqrt_out_final;
                    div_valid_in          <= div_valid_in_final;
                    mode_div              <= 2'b10;
                    
                    // 转BF16输出
                    fp32_2_bf16_in        <= div_out;
                    fp32_2_bf16_valid     <= div_valid;
                    out_data              <= out_w;
                    out_valid             <= out_valid_w;
                end
                
               silu_mode: begin
                   // ===== SiLU数据流 =====
                   // 公式：SiLU(x) = x / (1 + exp(-x))
                   // = x * sigmoid(x)
                   
                   // BRAM1存储原始输入x（用于最后的除法）
                   wea1                  <= fp32_valid_in;
                   ena1                  <= fp32_valid_in;
                   dina1                 <= fp32_data_in;
                   enb1                  <= add_valid;        // 当1+exp(-x)计算完成时读取x
                   
                   // 保存输入数据副本（用于子模块计算-x）
                   fp32_data_in_silu     <= fp32_data_in;
                   fp32_valid_in_silu    <= fp32_valid_in;
                   
                   // 计算exp(-x)
                   exp_in                <= inv_out;          // inv_out = -x（来自子模块）
                   exp_valid_in          <= inv_valid;
                   
                   // 计算1 + exp(-x)
                   add_in1               <= exp_out;
                   add_in3               <= 32'h3F800000;     // FP32的1.0
                   add_valid_in          <= exp_valid;
                   add_out_silu          <= add_out;
                   add_out_final         <= add_out_silu;
                   mode_add              <= 2'b10;            // 向量+常数
                   div_valid_in_final    <= enb1;
                   
                   // 最终除法x / (1 + exp(-x))
                   div_in1               <= doutb1;           // 原始x从BRAM1读出
                   div_in2               <= 32'b0;            // 未使用
                   div_in3               <= add_out_final;    // 1 + exp(-x)
                   div_valid_in          <= div_valid_in_final;
                   mode_div              <= 2'b01;            // 向量÷向量
                   
                   // 转BF16输出
                   fp32_2_bf16_in        <= div_out;
                   fp32_2_bf16_valid     <= div_valid;
                   out_data              <= out_w;
                   out_valid             <= out_valid_w;
               end
               
               el_add_mode: begin
                   // ===== 逐元素加法数据流 =====
                   // 特点：利用BRAM3的4:1宽度比，一次处理4组数据
                   // 输入：连续写入3072组512bit数据
                   // 输出：每次读出2048bit（4组512bit），两两相加得到2组1024bit
                   
                   // 写入BRAM3（写端口512bit）
                   wea3                  <= data_valid;
                   ena3                  <= data_valid;
                   dina3                 <= data_in;

                   // 读取BRAM3（读端口2048bit，一次读4组）
                   enb3                  <= start || ((addrb3 != 10'd767) && addrb3) || start_el_add;
                   start_el_add          <= start;
                   
                   // 将2048bit数据分成4路512bit，分别转FP32
                   data_in_1             <= doutb3[511:0];    // 第1组
                   data_valid_1_final    <= enb3;
                   data_valid_1          <= data_valid_1_final;

                   data_in_2             <= doutb3[1023:512]; // 第2组
                   data_valid_2_final    <= enb3;
                   data_valid_2          <= data_valid_2_final;

                   data_in_3             <= doutb3[1535:1024]; // 第3组
                   data_valid_3_final    <= enb3;
                   data_valid_3          <= data_valid_3_final;

                   data_in_4             <= doutb3[2047:1536]; // 第4组
                   data_valid_4_final    <= enb3;
                   data_valid_4          <= data_valid_4_final;
                   
                   // 加法器0计算data1 + data2
                   mode_add              <= 2'b01;            // 向量+向量
                   add_in1               <= fp32_data_in_1;
                   add_in2               <= fp32_data_in_2;
                   add_valid_in          <= fp32_valid_in_1;

                   // 加法器1计算data3 + data4
                   mode_add_1            <= 2'b01;
                   add_in1_1             <= fp32_data_in_3;
                   add_in2_1             <= fp32_data_in_4;
                   add_valid_in_1        <= fp32_valid_in_3;
                   
                   // 两个加法结果分别转BF16，拼接成1024bit输出
                   fp32_2_bf16_in        <= add_out;
                   fp32_2_bf16_valid     <= add_valid;
                   fp32_2_bf16_in_1      <= add_out_1;
                   fp32_2_bf16_valid_1   <= add_valid_1;
                   out_data_el           <= {out_w_1, out_w}; // 拼接：[511:0]=out_w, [1023:512]=out_w_1
                   out_valid             <= out_valid_w && out_valid_w_1; // 两路都有效时输出
               end
               
               el_mul_mode: begin
                   // ===== 逐元素乘法数据流 =====
                   // 流程与逐元素加法完全相同，只是将加法器替换为乘法器
                   
                   // 写BRAM3
                   wea3                  <= data_valid;
                   ena3                  <= data_valid;
                   dina3                 <= data_in;

                   // 读BRAM3
                   enb3                  <= start || ((addrb3 != 10'd767) && addrb3) || start_el_mul;
                   start_el_mul          <= start;
                   
                   // 分配4路数据
                   data_in_1             <= doutb3[511:0];
                   data_valid_1_final    <= enb3;
                   data_valid_1          <= data_valid_1_final;

                   data_in_2             <= doutb3[1023:512];
                   data_valid_2_final    <= enb3;
                   data_valid_2          <= data_valid_2_final;

                   data_in_3             <= doutb3[1535:1024];
                   data_valid_3_final    <= enb3;
                   data_valid_3          <= data_valid_3_final;

                   data_in_4             <= doutb3[2047:1536];
                   data_valid_4_final    <= enb3;
                   data_valid_4          <= data_valid_4_final;
                   
                   // 乘法器0：data1 * data2
                   mode_mul              <= 2'b01;            // 向量×向量
                   mul_in1               <= fp32_data_in_1;
                   mul_in2               <= fp32_data_in_2;
                   mul_valid_in          <= fp32_valid_in_1;

                   // 乘法器1：data3 * data4
                   mode_mul_1            <= 2'b01;
                   mul_in1_1             <= fp32_data_in_3;
                   mul_in2_1             <= fp32_data_in_4;
                   mul_valid_in_1        <= fp32_valid_in_3;
                   
                   // 转BF16拼接输出
                   fp32_2_bf16_in        <= mul_out;
                   fp32_2_bf16_valid     <= mul_valid;
                   fp32_2_bf16_in_1      <= mul_out_1;
                   fp32_2_bf16_valid_1   <= mul_valid_1;
                   out_data_el           <= {out_w_1, out_w};
                   out_valid             <= out_valid_w && out_valid_w_1;
               end
               
               gelu_mode:begin
                   // ===== GELU数据流 =====
                   // 近似公式：GELU(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
                   // 实现策略：分段计算多项式，使用exp做tanh近似
                   
                   // BRAM1和BRAM2同时存储输入（用于多次读取）
                   wea1                  <= fp32_valid_in;
                   ena1                  <= fp32_valid_in;
                   dina1                 <= fp32_data_in;
                   wea2                  <= fp32_valid_in;
                   ena2                  <= fp32_valid_in;
                   dina2                 <= fp32_data_in;
                   
                   // BRAM读控制
                   enb1                  <= mul_valid_in_gelu; // 读x用于乘以系数
                   enb2                  <= exp_valid;         // 读x用于最终乘法
                   
                   // 输入数据准备
                   fp32_data_in_gelu     <= fp32_data_in;
                   fp32_valid_in_gelu    <= fp32_valid_in;
                   
                   // 乘法器0：计算多项式项（x², x³等）
                   mode_mul              <= 2'b01;
                   mul_in1               <= mul_in1_gelu;     // 来自gelu子模块
                   mul_in2               <= mul_in2_gelu;
                   mul_valid_in          <= mul_valid_in_gelu;
                   mul_valid_gelu        <= mul_valid;
                   mul_out_gelu          <= mul_out;
                   mul_valid_final       <= mul_valid_gelu;
                   mul_out_final         <= mul_out_gelu;
                   
                   // 乘法器1：乘以常数√(2/π)
                   mode_mul_1            <= 2'b10;
                   mul_in1_1             <= doutb1;
                   mul_in3_1             <= 32'h3FCC422A;     // √(2/π) ≈ 0.7978845608
                   mul_valid_in_1_final  <= mul_valid_in_gelu;
                   mul_valid_in_1_final1 <= mul_valid_in_1_final; // 两级流水线延迟
                   mul_valid_in_1        <= mul_valid_in_1_final1;

                   // 乘法器2：最终乘法（x与tanh项相乘）
                   mode_mul_2            <= 2'b01;
                   mul_in1_2             <= doutb2;
                   mul_in2_2             <= exp_out_final_mul;
                   mul_valid_in_2        <= exp_valid_final;
                   
                   // 4级流水线寄存器（用于时序对齐，因为GELU计算路径较长）
                   mul_out_2_0           <= mul_out_2;
                   mul_valid_2_0         <= mul_valid_2;
                   mul_out_2_1           <= mul_out_2_0;
                   mul_valid_2_1         <= mul_valid_2_0;
                   mul_out_2_2           <= mul_out_2_1;
                   mul_valid_2_2         <= mul_valid_2_1;
                   mul_out_2_3           <= mul_out_2_2;
                   mul_valid_2_3         <= mul_valid_2_2;
                   
                   // 加法器0：多项式项相加（x + 0.044715*x³）
                   mode_add              <= 2'b01;
                   add_in1               <= mul_out_final;
                   add_in2               <= mul_out_1;
                   add_valid_in          <= mul_valid_1;

                   // 加法器1：tanh近似中的+1项
                   mode_add_1            <= 2'b10;
                   add_in1_1             <= exp_out_final_add;
                   add_in3_1             <= 32'h3f800000;     // 1.0
                   add_valid_in_1        <= exp_valid_final;
                   
                   // 指数运算（用于tanh近似）
                   exp_in                <= add_out;
                   exp_valid_in          <= add_valid;
                   exp_out_gelu          <= exp_out;
                   exp_valid_gelu        <= exp_valid;
                   
                   // 最终除法（完成归一化）
                   mode_div              <= 2'b01;            // 向量÷向量
                   div_in1_gelu          <= mul_out_2_3;
                   div_valid_in_gelu     <= mul_valid_2_3;
                   div_in1               <= div_in1_gelu;
                   div_in3               <= add_out_1;
                   div_valid_in          <= div_valid_in_gelu;
                   
                   // 转BF16输出
                   fp32_2_bf16_in        <= div_out;
                   fp32_2_bf16_valid     <= div_valid;
                   out_data              <= out_w;
                   out_valid             <= out_valid_w;                    
               end
               
               default: begin
                   // 默认状态：清零所有控制信号
                   out_valid          <= 1'b0;
                   add_valid_in       <= 1'b0;
                   sub_valid_in       <= 1'b0;
                   mul_valid_in       <= 1'b0;
                   exp_valid_in       <= 1'b0;
                   sum_valid_in       <= 1'b0;
                   sum_valid_in_1     <= 1'b0;
                   div_valid_in       <= 1'b0;
                   fp32_2_bf16_valid  <= 1'b0;
                   fp32_2_bf16_valid_1<= 1'b0;
                   add_valid1         <= 1'b0;
                   el_add_cnt         <= 2'b0;
                   el_cnt_flag        <= 1'b0;
                   el_mul_cnt         <= 2'b0;
                   mode_add           <= 2'b00;
                   mode_add_1         <= 2'b00;
                   mode_div           <= 2'b00;
                   mode_mul           <= 2'b00;
                   mode_mul_1         <= 2'b00;
                   mode_mul_2         <= 2'b00;
                   wea1               <= 1'b0;
                   ena1               <= 1'b0;
                   enb1               <= 1'b0;
                   wea2               <= 1'b0;
                   ena2               <= 1'b0;
                   enb2               <= 1'b0;                             
                   wea3               <= 1'b0;
                   ena3               <= 1'b0;
                   enb3               <= 1'b0;                            
               end
           endcase
       end
   end
   
   // =================================
   // 完成信号生成逻辑
   // 功能：在输出有效后的下一拍产生单周期done脉冲
   // =================================
   always @(posedge clk or posedge rst) begin
       if(rst) begin
           done      <= 1'b0;
           done_flag <= 1'b0;
       end else if(out_valid)begin
           done_flag <= 1'b1;           // 输出有效时标记
       end
       else begin
           done      <= done_flag;      // 延迟一拍输出done
           done_flag <= 1'b0;           // 清除标志
       end
   end

   // =================================
   // 子模块实例化
   // =================================
   
   // Softmax控制模块：管理最大值查找和触发时序
   softmax_top u_softmax_top (
       .clk(clk),
       .rst(rst),
       .softmax_data_in(softmax_data_in),
       .data_valid(softmax_data_valid),
       .max_valid(max_valid_softmax),
       .max_val_fp32(max_val_fp32_softmax),
       .fp32_data(fp32_data_softmax),
       .sum_valid(sum_valid_softmax),
       .div_valid_in_final(div_valid_in_softmax)
   );

   // LayerNorm控制模块：管理方差计算（σ²+ε）
   layernorm_top u_layernorm_top (
       .clk(clk),
       .rst(rst),
       .sum_out(sum_out_layernorm),
       .sum_out_valid(sum_out_valid_layernorm),
       .add_out(add_out_layernorm),
       .add_out_valid(add_out_valid_layernorm)
   );

   // RMSNorm控制模块：管理均方根计算
   RMSnorm_top u_RMSnorm_top (
       .clk(clk),
       .rst(rst),
       .sum_out(sum_out_rmsnorm),
       .sum_valid(sum_valid_rmsnorm),
       .add_out(add_out_rmsnorm),
       .add_out_valid(add_out_valid_rmsnorm)
   );

   // SiLU控制模块：提供取反运算（-x）
   SiLU_top u_SiLU_top (
       .clk(clk),
       .rst(rst),
       .fp32_data_in(fp32_data_in_silu),
       .fp32_valid_in(fp32_valid_in_silu),
       .inv_out(inv_out),
       .inv_valid(inv_valid)
   );

   // GELU控制模块：管理多项式计算和流水线
   gelu u_gelu (
       .clk(clk),
       .rst(rst),
       .x_data_in(fp32_data_in_gelu),
       .data_valid(fp32_valid_in_gelu),
       .mul_out_0(mul_in1_gelu),
       .mul_out_1(mul_in2_gelu),
       .mul_valid(mul_valid_in_gelu),
       .exp_out(exp_out_gelu), 
       .exp_valid(exp_valid_gelu),
       .exp_out_final_mul(exp_out_final_mul),
       .exp_out_final_add(exp_out_final_add),
       .exp_valid_final(exp_valid_final)
   );

   // BF16到FP32转换器（主输入通道）
   bf16_2_fp32 u_bf16_to_fp32 (
       .clk(clk),
       .rst(rst),
       .data_valid(data_valid),
       .bf16_in(data_in),
       .fp32_out(fp32_data_in),
       .fp32_valid(fp32_valid_in)
   );

   // BF16到FP32转换器1-4（逐元素操作的4路并行通道）
   bf16_2_fp32 u_bf16_to_fp32_1 (
       .clk(clk),
       .rst(rst),
       .data_valid(data_valid_1),
       .bf16_in(data_in_1),
       .fp32_out(fp32_data_in_1),
       .fp32_valid(fp32_valid_in_1)
   );

   bf16_2_fp32 u_bf16_to_fp32_2 (
       .clk(clk),
       .rst(rst),
       .data_valid(data_valid_2),
       .bf16_in(data_in_2),
       .fp32_out(fp32_data_in_2),
       .fp32_valid(fp32_valid_in_2)
   );

   bf16_2_fp32 u_bf16_to_fp32_3 (
       .clk(clk),
       .rst(rst),
       .data_valid(data_valid_3),
       .bf16_in(data_in_3),
       .fp32_out(fp32_data_in_3),
       .fp32_valid(fp32_valid_in_3)
   );

   bf16_2_fp32 u_bf16_to_fp32_4 (
       .clk(clk),
       .rst(rst),
       .data_valid(data_valid_4),
       .bf16_in(data_in_4),
       .fp32_out(fp32_data_in_4),
       .fp32_valid(fp32_valid_in_4)
   );

   // 加法器0（支持向量+向量、向量+常数）
   add u_add (
       .clk(clk),
       .rst(rst),
       .add_in1(add_in1),
       .add_in2(add_in2),
       .add_in3(add_in3),
       .data_valid(add_valid_in),
       .mode(mode_add),
       .add_out(add_out),
       .add_valid(add_valid)
   );

   // 加法器1
   add u_add_1 (
       .clk(clk),
       .rst(rst),
       .add_in1(add_in1_1),
       .add_in2(add_in2_1),
       .add_in3(add_in3_1),
       .data_valid(add_valid_in_1),
       .mode(mode_add_1),
       .add_out(add_out_1),
       .add_valid(add_valid_1)
   );

   // 减法器（向量-标量）
   sub u_sub (
       .clk(clk),
       .rst(rst),
       .sub_in1(fp32_data),
       .sub_in2(sub_in),
       .data_valid(sub_valid_in),
       .sub_out(sub_out),
       .sub_valid(sub_valid)
   );

   // 乘法器0-2
   mul u_mul (
       .clk(clk),
       .rst(rst),
       .mode(mode_mul),
       .mul_in1(mul_in1),
       .mul_in2(mul_in2),
       .mul_in3(mul_in3),
       .data_valid(mul_valid_in),
       .mul_out(mul_out),
       .mul_valid(mul_valid)
   );

   mul u_mul_1 (
       .clk(clk),
       .rst(rst),
       .mode(mode_mul_1),
       .mul_in1(mul_in1_1),
       .mul_in2(mul_in2_1),
       .mul_in3(mul_in3_1),
       .data_valid(mul_valid_in_1),
       .mul_out(mul_out_1),
       .mul_valid(mul_valid_1)
   );

   mul u_mul_2 (
       .clk(clk),
       .rst(rst),
       .mode(mode_mul_2),
       .mul_in1(mul_in1_2),
       .mul_in2(mul_in2_2),
       .mul_in3(mul_in3_2),
       .data_valid(mul_valid_in_2),
       .mul_out(mul_out_2),
       .mul_valid(mul_valid_2)
   );

   // 指数运算模块（基于查找表或CORDIC算法）
   exp u_exp (
       .clk(clk),
       .rst(rst),
       .exp_in(exp_in),
       .data_valid(exp_valid_in),
       .exp_out(exp_out),
       .exp_valid(exp_valid)
   );

   // 累加器0（树形加法器，将32个FP32累加为1个标量）
   acc_sum768 u_add_sigma (
       .clk(clk),
       .rst(rst),
       .sum_in(sum_in),
       .data_valid(sum_valid_in),
       .sum_total(sum_out),
       .sum_total_valid(sum_valid)
   );

   // 累加器1
   acc_sum768 u_add_sigma_1 (
       .clk(clk),
       .rst(rst),
       .sum_in(sum_in_1),
       .data_valid(sum_valid_in_1),
       .sum_total(sum_out_1),
       .sum_total_valid(sum_valid_1)
   );

   // 除法器（基于Newton-Raphson迭代或查找表）
   div u_div (
       .clk(clk),
       .rst(rst),
       .div_in1(div_in1),
       .div_in2(div_in2),
       .mode(mode_div),
       .div_in3(div_in3),
       .data_valid(div_valid_in),
       .div_out(div_out),
       .div_valid(div_valid)
   );

   // 平方根模块（Newton-Raphson或CORDIC算法）
   sqrt u_sqrt (
       .clk(clk),
       .rst(rst),
       .data_in(sqrt_in),
       .data_valid(sqrt_valid_in),
       .data_out(sqrt_out),
       .sqrt_valid(sqrt_valid)
   );

   // FP32到BF16转换器0（舍入和截断）
   fp32_to_bf16 u_fp32_to_bf16 (
       .clk(clk),
       .rst(rst),
       .data_in(fp32_2_bf16_in),
       .data_valid(fp32_2_bf16_valid),
       .data_out_valid(out_valid_w), 
       .data_out(out_w)
   );

   // FP32到BF16转换器1
   fp32_to_bf16 u_fp32_to_bf16_1 (
       .clk(clk),
       .rst(rst),
       .data_in(fp32_2_bf16_in_1),
       .data_valid(fp32_2_bf16_valid_1),
       .data_out_valid(out_valid_w_1), 
       .data_out(out_w_1)
   );

   // 计数器模块（统计start到done的周期数，用于性能分析）
   acc_counter u_acc_counter (
       .clk(clk),
       .rst(rst),
       .start(start),
       .done(done),
       .count(count)
   );

   // =================================
   // BRAM实例化和地址计数器
   // =================================
   
   // BRAM1：对称双端口RAM（1536×1024bit）
   blk_mem_gen_1 u_blk_mem_gen1 (
       .clka(clk),
       .ena(ena1),
       .wea(wea1),
       .addra(addra1),
       .dina(dina1),
       .clkb(clk),
       .enb(enb1),
       .addrb(addrb1),
       .doutb(doutb1)
   );
   
   // BRAM1写地址计数器（0-1535循环）
   always @(posedge clk or posedge rst) begin
       if (rst) begin
           addra1 <= 11'd0;
       end else if (ena1 && wea1) begin           
           if (addra1 == 11'd1535) begin
               addra1 <= 11'd0;         // 达到上限后归零
           end
           else begin
               addra1 <= addra1 + 1'b1;
           end
       end
       else begin
           addra1 <= addra1;            // 保持
       end
   end
   
   // BRAM1读地址计数器
   always @(posedge clk or posedge rst) begin
       if (rst) begin
           addrb1 <= 11'd0;
       end else if (enb1) begin           
           if (addrb1 == 11'd1535) begin
               addrb1 <= 11'd0;
           end
           else begin
               addrb1 <= addrb1 + 1'b1;
           end
       end
       else begin
           addrb1 <= addrb1;
       end
   end

   // BRAM2：对称双端口RAM（1536×1024bit）
   bram_dual_port u_blk_mem_gen2 (
       .clk(clk),
       .ena(ena2),
       .wea(wea2),
       .addra(addra2),
       .dina(dina2),
       .enb(enb2),
       .addrb(addrb2),
       .doutb(doutb2)
   );
   
   // BRAM2写地址计数器
   always @(posedge clk or posedge rst) begin
       if (rst) begin
           addra2 <= 11'd0;
       end else if (ena2 && wea2) begin           
           if (addra2 == 11'd1535) begin
               addra2 <= 11'd0;
           end
           else begin
               addra2 <= addra2 + 1'b1;
           end
       end
       else begin
           addra2 <= addra2;
       end
   end
   
   // BRAM2读地址计数器
   always @(posedge clk or posedge rst) begin
       if (rst) begin
           addrb2 <= 11'd0;
       end else if (enb2) begin           
           if (addrb2 == 11'd1535) begin
               addrb2 <= 11'd0;
           end
           else begin
               addrb2 <= addrb2 + 1'b1;
           end
       end
       else begin
           addrb2 <= addrb2;
       end
   end

   // BRAM3：非对称宽度双端口RAM
   // 特点：写入512bit，读出2048bit（4倍宽度）
   // 用途：逐元素操作中存储4组数据，实现并行处理
   blk_mem_gen_0 u_blk_mem_gen3 (
       .clka(clk),
       .wea(wea3),
       .addra(addra3),
       .dina(dina3),
       .ena(ena3),
       .clkb(clk),
       .enb(enb3),
       .addrb(addrb3),
       .doutb(doutb3)
   );
   
   // BRAM3写地址计数器（0-3071循环）
   // 说明：存储3072组512bit数据（相当于768组2048bit）
   always @(posedge clk or posedge rst) begin
       if (rst) begin
           addra3 <= 12'd0;
       end else if (ena3 && wea3) begin           
           if (addra3 == 12'd3071) begin
               addra3 <= 12'd0;         // 写满后循环
           end
           else begin
               addra3 <= addra3 + 1'b1;
           end
       end
       else begin
           addra3 <= addra3;
       end
   end
   
   // BRAM3读地址计数器（0-767循环）
   // 说明：每次读取2048bit（4组512bit），所以地址范围是写地址的1/4
   always @(posedge clk or posedge rst) begin
       if (rst) begin
           addrb3 <= 10'd0;
       end else if (enb3) begin           
           if (addrb3 == 10'd767) begin
               addrb3 <= 10'd0;         // 读满后循环
           end
           else begin
               addrb3 <= addrb3 + 1'b1;
           end
       end
       else begin
           addrb3 <= addrb3;
       end
   end

endmodule


                
