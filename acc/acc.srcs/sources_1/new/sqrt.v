`timescale 1ns / 1ps

module sqrt(
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0]    data_in,        
    input  wire        data_valid,
    // 输出寄存，便于在有效时刻一次性采样并保持稳定
    output reg  [31:0]  data_out,
    output reg         sqrt_valid
);
    wire         sqrt_valid_temp;
    // 来自IP的乘法结果（组合/直连），在有效沿时再寄存到mul_out
    wire [31:0]  sqrt_out_w;
    // =================================
    // 32 路浮点平方根 IP 实例化（每路 sqrt(a[i])）
    // =================================
    floating_point_5 u_fp32_sqrt (
        .aclk(clk),
        .s_axis_a_tvalid(data_valid),
        .s_axis_a_tdata(data_in), // 输入，FP32
        .m_axis_result_tvalid(sqrt_valid_temp),
        .m_axis_result_tdata(sqrt_out_w)  // 结果为 FP32
    );
    // =================================
    // 乘法结果有效性判断
    // =================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sqrt_valid <= 1'b0;
        end else if (sqrt_valid_temp) begin
            // 当32路均有效时，打一拍输出并拉高sqrt_valid（1拍脉冲）
            sqrt_valid <= 1'b1;
            data_out   <= sqrt_out_w;
        end else begin
            sqrt_valid <= 1'b0;
            data_out   <= data_out;
        end
    end
endmodule
