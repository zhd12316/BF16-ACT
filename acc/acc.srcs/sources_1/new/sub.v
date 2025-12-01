`timescale 1ns / 1ps



module sub(
    input  wire        clk,
    input  wire        rst,
    input  wire [1023:0]  sub_in1,
    input  wire [31:0]   sub_in2,
    input  wire        data_valid,
    output reg  [1023:0]  sub_out,
    output reg         sub_valid
    );
    wire [1023:0] sub_out_temp;
    wire [31:0]   sub_valid_temp;
    reg  [31:0]   sub_in2_reg;
    reg  [1023:0] sub_in1_reg;
    reg           data_valid_reg;
    reg  [1023:0] sub_in1_reg2;
    reg  [31:0]   sub_in2_reg2;
    reg           data_valid_reg2;    
    always @(posedge clk) begin
        sub_in1_reg <= sub_in1;
        sub_in2_reg <= sub_in2;
        data_valid_reg <= data_valid;
    end

    always @(posedge clk) begin
        sub_in1_reg2 <= sub_in1_reg;
        sub_in2_reg2 <= sub_in2_reg;
        data_valid_reg2 <= data_valid_reg;
    end

    // =================================
    // 32 路浮点减法 IP 实例化
    // =================================
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin: sub_module
            floating_point_1 u_fp32_sub (
                .aclk(clk),
                .s_axis_a_tvalid(data_valid_reg2),
                .s_axis_a_tdata(sub_in1_reg2[i*32 +: 32]), 
                .s_axis_b_tvalid(data_valid_reg2),
                .s_axis_b_tdata(sub_in2_reg2),      
                .m_axis_result_tvalid(sub_valid_temp[i]),
                .m_axis_result_tdata(sub_out_temp[i*32 +: 32])  // 结果为 FP32
            );
        end        
    endgenerate
    // =================================
    // 减法结果有效性判断
    // =================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sub_valid <= 1'b0;
        end else if (&sub_valid_temp) begin
            sub_valid <= 1'b1;
            sub_out <= sub_out_temp;
        end else begin
            sub_valid <= 1'b0;
            sub_out <= sub_out;
        end
    end

endmodule
