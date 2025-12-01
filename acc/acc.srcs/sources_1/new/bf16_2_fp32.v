`timescale 1ns / 1ps
    

module bf16_2_fp32(
    input  wire        clk,
    input  wire        rst,
    input  wire [511:0]  bf16_in,
    input  wire        data_valid,
    output reg  [1023:0]  fp32_out,
    output reg         fp32_valid
);
    wire [1023:0] fp32_out_temp;
    // 将 32 路 BF16 转换为 32 路 FP32：
    // bf16: {sign[15], exp[14:7] (8b), frac[6:0] (7b)}
    // fp32: {sign[31], exp[30:23] (8b), frac[22:0] (23b)}
    // 直接映射：sign 原样；exp 原样；frac 放在高 7 位，低 16 位补 0。
    // 注：该位拼接方式符合 IEEE-754 语义，能够保留 BF16 次正规数（exp=0, frac!=0），不会冲零。
    // 组合转换结果
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : bf16_to_fp32_module
            assign fp32_out_temp[i*32 +: 32] = {
                bf16_in[i*16 + 15],                    // sign
                bf16_in[i*16 + 14 : i*16 + 7],         // exponent (8 bits)
                bf16_in[i*16 + 6  : i*16 + 0],         // fraction high 7 bits
                16'b0                             // pad low 16 bits
            };
        end
    endgenerate

    // 输出寄存
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            fp32_valid <= 1'b0;
        end else if (data_valid) begin
            fp32_out   <= fp32_out_temp;
            fp32_valid <= 1'b1;
        end else begin
            fp32_out   <= fp32_out;
            fp32_valid <= 1'b0;
        end
    end

endmodule
