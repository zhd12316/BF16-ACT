module add_sigma (
    input  wire        clk,
    input  wire        rst,
    input  wire [1023:0] data_in,         // 32 x fp32
    input  wire        data_valid,  // 输入有效
    output reg  [31:0] sum_out,     // 输出累加结果（fp32）
    output reg         sum_valid     // 输出有效标志
);

    // =================================
    // 加法树：32 -> 16 -> 8 -> 4 -> 2 -> 1
    // 使用相同的浮点加法 IP（floating_point_0），各层 tvalid 由上一层 all-valid 触发
    // =================================

    // 第1层：32 -> 16（这里生成16个加法器，按 0..31 对称配对）
    wire [31:0] sum_1 [15:0];
    wire [15:0] sum_valid_1;
    wire        sum_valid_1all;
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : add_l1
            floating_point_0 u_add_l1 (
                .aclk(clk),
                .s_axis_a_tvalid(data_valid),
                .s_axis_a_tdata(data_in[i*32 +: 32]),
                .s_axis_b_tvalid(data_valid),
                .s_axis_b_tdata(data_in[(31 - i)*32 +: 32]),
                .m_axis_result_tvalid(sum_valid_1[i]),
                .m_axis_result_tdata(sum_1[i])
            );
        end
    endgenerate
    assign sum_valid_1all = &sum_valid_1;

    // 第2层：16 -> 8
    wire [31:0] sum_2 [7:0];
    wire [7:0] sum_valid_2;
    wire        sum_valid_2all;
    generate
        for (i = 0; i < 8; i = i + 1) begin : add_l2
            floating_point_0 u_add_l2 (
                .aclk(clk),
                .s_axis_a_tvalid(sum_valid_1all),
                .s_axis_a_tdata(sum_1[i]),
                .s_axis_b_tvalid(sum_valid_1all),
                .s_axis_b_tdata(sum_1[15 - i]),
                .m_axis_result_tvalid(sum_valid_2[i]),
                .m_axis_result_tdata(sum_2[i])
            );
        end
    endgenerate
    assign sum_valid_2all = &sum_valid_2;

    // 第3层：8 -> 4
    wire [31:0] sum_3 [3:0];
    wire [3:0]  sum_valid_3;
    wire        sum_valid_3all;
    generate
        for (i = 0; i < 4; i = i + 1) begin : add_l3
            floating_point_0 u_add_l3 (
                .aclk(clk),
                .s_axis_a_tvalid(sum_valid_2all),
                .s_axis_a_tdata(sum_2[i]),
                .s_axis_b_tvalid(sum_valid_2all),
                .s_axis_b_tdata(sum_2[7 - i]),
                .m_axis_result_tvalid(sum_valid_3[i]),
                .m_axis_result_tdata(sum_3[i])
            );
        end
    endgenerate
    assign sum_valid_3all = &sum_valid_3;

    // 第4层：4 -> 2
    wire [31:0] sum_4 [1:0];
    wire [1:0]  sum_valid_4;
    wire        sum_valid_4all;
    generate
        for (i = 0; i < 2; i = i + 1) begin : add_l4
            floating_point_0 u_add_l4 (
                .aclk(clk),
                .s_axis_a_tvalid(sum_valid_3all),
                .s_axis_a_tdata(sum_3[i]),
                .s_axis_b_tvalid(sum_valid_3all),
                .s_axis_b_tdata(sum_3[3 - i]),
                .m_axis_result_tvalid(sum_valid_4[i]),
                .m_axis_result_tdata(sum_4[i])
            );
        end
    endgenerate
    assign sum_valid_4all = &sum_valid_4;

    // 第5层：2 -> 1
    wire [31:0] sum_5;
    wire        sum_valid_5;
    floating_point_0 u_add_l5 (
        .aclk(clk),
        .s_axis_a_tvalid(sum_valid_4all),
        .s_axis_a_tdata(sum_4[0]),
        .s_axis_b_tvalid(sum_valid_4all),
        .s_axis_b_tdata(sum_4[1]),
        .m_axis_result_tvalid(sum_valid_5),
        .m_axis_result_tdata(sum_5)
    );


    // 输出寄存
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sum_out   <= 32'b0;
            sum_valid <= 1'b0;
        end else if (sum_valid_5) begin
            sum_out   <= sum_5;
            sum_valid <= 1'b1;
        end else begin
            sum_valid <= 1'b0;
            sum_out   <= sum_out;
        end
    end

endmodule
