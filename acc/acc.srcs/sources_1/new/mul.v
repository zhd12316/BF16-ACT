`timescale 1ns / 1ps

module mul(
    input  wire        clk,
    input  wire        rst,
    input  wire [1:0]  mode,
    input  wire [1023:0]  mul_in1,        // Numerator: 32x FP32 (e.g., exp values)
    input  wire [31:0]    mul_in3,        // Denominator: scalar FP32 (e.g., total sum)
    input  wire [1023:0]  mul_in2,
    input  wire        data_valid,
    // 输出寄存，便于在有效时刻一次性采样并保持稳定
    output reg  [1023:0]  mul_out,
    output reg         mul_valid
);
    wire [31:0] mul_valid_temp;
    // 来自IP的乘法结果（组合/直连），在有效沿时再寄存到mul_out
    wire [1023:0] mul_out_w;
    wire [1023:0] mul_in;
    // =================================
    // 32 路浮点乘法 IP 实例化（每路 a[i] * b[i]）
    // =================================
    genvar i;
    generate
        for ( i = 0; i < 32; i = i + 1 ) begin:mul_module
            assign mul_in[i*32 +: 32] = (mode == 2'b10) ? mul_in3 : mul_in2[i*32 +: 32];
            floating_point_4 u_fp32_mul (
                .aclk(clk),
                .s_axis_a_tvalid(data_valid),
                .s_axis_a_tdata(mul_in1[i*32 +: 32]), // 被乘数，FP32
                .s_axis_b_tvalid(data_valid),
                .s_axis_b_tdata(mul_in[i*32 +: 32]),       // 除数为总和 sum_val，FP32
                .m_axis_result_tvalid(mul_valid_temp[i]), 
                .m_axis_result_tdata(mul_out_w[i*32 +: 32])  // 结果为 FP32
            );
        end
    endgenerate
    // =================================
    // 乘法结果有效性判断
    // =================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mul_valid <= 1'b0;
        end else if (&mul_valid_temp) begin
            // 当32路均有效时，打一拍输出并拉高mul_valid（1拍脉冲）
            mul_valid <= 1'b1;
            mul_out   <= mul_out_w;
        end else begin
            mul_valid <= 1'b0;
            mul_out   <= mul_out;
        end
    end
endmodule
