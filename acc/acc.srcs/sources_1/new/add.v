`timescale 1ns / 1ps



module add(
    input  wire        clk,
    input  wire        rst,
    input  wire [1023:0]  add_in1,
    input  wire [1023:0]  add_in2,
    input  wire [31:0]    add_in3,
    input  wire           data_valid,
    input  wire [1:0]     mode,
    output reg  [1023:0]  add_out,
    output reg            add_valid
);
    wire [1023:0] add_in;
    wire [1023:0] add_out_temp;
    wire [31:0] add_valid_temp;

    // =================================
    // 32 路浮点加法 IP 实例化
    // =================================
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin: add_module
            assign add_in[i*32 +: 32] = (mode == 2'b10) ? add_in3 : add_in2[i*32 +: 32];
            floating_point_0 u_fp32_add (
                .aclk(clk),
                .s_axis_a_tvalid(data_valid),
                .s_axis_a_tdata(add_in1[i*32 +: 32]), 
                .s_axis_b_tvalid(data_valid),
                .s_axis_b_tdata(add_in[i*32 +: 32]),      
                .m_axis_result_tvalid(add_valid_temp[i]),
                .m_axis_result_tdata(add_out_temp[i*32 +: 32])  // 结果为 FP32
            );
        end        
    endgenerate
    // =================================
    // 加法结果有效性判断
    // =================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            add_valid <= 1'b0;
        end else if (&add_valid_temp) begin
            add_valid <= 1'b1;
            add_out <= add_out_temp;
        end else begin
            add_valid <= 1'b0;
            add_out <= add_out;
        end
    end

endmodule
