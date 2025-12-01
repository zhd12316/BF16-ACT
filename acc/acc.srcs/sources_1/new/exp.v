`timescale 1ns / 1ps


module exp(
    input  wire        clk,
    input  wire        rst,
    input  wire [1023:0]  exp_in,
    input  wire        data_valid,
    // 输出寄存，便于在有效时刻一次性采样并保持稳定
    output reg  [1023:0]  exp_out,
    output reg         exp_valid
    );
    wire [31:0] exp_valid_temp;
    // 来自IP的指数结果（组合/直连），在有效沿时再寄存到exp_out
    wire [1023:0] exp_out_w;
    // =================================
    // 32 路浮点指数 IP 实例化
    // =================================
    genvar i;
    generate
        for ( i = 0; i < 32; i = i + 1 ) begin:exp_module
            floating_point_2 u_fp32_exp1 (
                .aclk(clk),
                .s_axis_a_tvalid(data_valid),
                .s_axis_a_tdata(exp_in[i*32 +: 32]), // 输入为 FP32
                .m_axis_result_tvalid(exp_valid_temp[i]), 
                .m_axis_result_tdata(exp_out_w[i*32 +: 32])  // 结果为 FP32
            );
        end
    endgenerate
    // =================================
    // 指数结果有效性判断
    // =================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_valid <= 1'b0;
        end else if (&exp_valid_temp) begin
            // 当32路均有效时，打一拍输出并拉高exp_valid（1拍脉冲）
            exp_valid <= 1'b1;
            exp_out   <= exp_out_w;
        end else begin
            exp_valid <= 1'b0;
            exp_out   <= exp_out;
        end
    end
endmodule
