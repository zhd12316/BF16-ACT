`timescale 1ns / 1ps

module div(
    input  wire        clk,
    input  wire        rst,
    input  wire [1:0]  mode,
    input  wire [1023:0]  div_in1,        // Numerator: 32x FP32 (e.g., exp values)
    input  wire [31:0]    div_in2,  // Denominator: scalar FP32 (e.g., total sum)
    input  wire [1023:0]  div_in3,
    input  wire        data_valid,
    // 输出寄存，便于在有效时刻一次性采样并保持稳定
    output reg  [1023:0]  div_out,
    output reg         div_valid
);
    wire [31:0] div_valid_temp;
    reg          div_valid_out_w1;
    wire [1023:0] div_out_w;    
    // 来自IP的除法结果（组合/直连），在有效沿时再寄存到div_out
    wire [767:0] div_out_w1;
    reg  [767:0] div_out_w_reg;
    wire [767:0] div_in1_w;
    wire [767:0] div_in;
    reg  [767:0] div_in1_w1;
    reg  [767:0] div_in_w1;
    reg          div_valid_in_w1;
    // =================================
    // 32 路浮点除法 IP 实例化（每路 a[i] / sum_val）
    // =================================
    genvar i;
    generate
        for ( i = 0; i < 32; i = i + 1 ) begin:div_module_before
            assign div_in[i*24 +: 24] = (mode == 2'b10) ? div_in2[31:8] : div_in3[(i*32 + 31) -: 24];
            assign div_in1_w[i*24 +: 24] = div_in1[(i*32 + 31) -: 24];
        end 
    endgenerate

    always @(posedge clk) begin
        if (data_valid) begin
            div_in1_w1 <= div_in1_w;
            div_in_w1 <= div_in;
            div_valid_in_w1 <= 1'b1;
        end
        else begin
            div_valid_in_w1 <= 1'b0;
        end
    end

    generate
        for ( i = 0; i < 32; i = i + 1 ) begin:div_module
            floating_point_3 u_fp32_div (
                .aclk(clk),
                .s_axis_a_tvalid(div_valid_in_w1),
                .s_axis_a_tdata(div_in1_w1[i*24 +: 24]), // 被除 数，FP32
                .s_axis_b_tvalid(div_valid_in_w1),
                .s_axis_b_tdata(div_in_w1[i*24 +: 24]),       // 除数为总和 sum_val，FP32
                .m_axis_result_tvalid(div_valid_temp[i]), 
                .m_axis_result_tdata(div_out_w1[i*24 +: 24])  // 结果为 FP32
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (&div_valid_temp) begin
            div_out_w_reg <= div_out_w1;
            div_valid_out_w1 <= 1'b1;
        end else begin
            div_valid_out_w1 <= 1'b0;
        end
    end

    generate
        for ( i = 0; i < 32; i = i + 1 ) begin:div_module_after
            assign div_out_w[i*32 +: 32] = {div_out_w_reg[i*24 +: 24],8'b0};
        end         
    endgenerate
    // =================================
    // 除法结果有效性判断
    // =================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            div_valid <= 1'b0;
        end else if (div_valid_out_w1) begin
            // 当32路均有效时，打一拍输出并拉高div_valid（1拍脉冲）
            div_valid <= 1'b1;
            div_out   <= div_out_w;
        end else begin
            div_valid <= 1'b0;
        end
    end
endmodule
