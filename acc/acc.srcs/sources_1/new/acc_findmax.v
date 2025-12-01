`timescale 1ns / 1ps


module acc_findmax(
    input  wire        clk,
    input  wire        rst,
    input  wire [511:0]  data_in,
    input  wire        data_valid,
    output reg  [15:0]  max_out,
    output reg         max_valid
    );
    //****************************************
    // 分级比较寻找最大值模块设计
    //****************************************
    //第一轮最大值寻找
    genvar i;
    wire [15:0]  max_2 [15:0];
    wire [15:0] max_valid_2;
    wire max_valid_2all;
    generate
        for (i = 0; i < 16; i = i + 1) begin : find_max_2
            acc_comparison u_acc_comparison_2 (
                .clk(clk),
                .rst(rst),
                .a(data_in[i*16 +: 16]),
                .b(data_in[(31-i)*16 +: 16]),
                .data_valid(data_valid),
                .max_out(max_2[i]),
                .max_valid(max_valid_2[i])
            );
        end
    endgenerate
    assign max_valid_2all = &max_valid_2;
    //第二轮最大值寻找
    wire [15:0]  max_3 [7:0];
    wire [7:0] max_valid_3;
    wire max_valid_3all;
    generate
        for (i = 0; i < 8; i = i + 1) begin : find_max_3
            acc_comparison u_acc_comparison_3 (
                .clk(clk),
                .rst(rst),
                .a(max_2[i]),
                .b(max_2[15 - i]),
                .data_valid(max_valid_2all),
                .max_out(max_3[i]),
                .max_valid(max_valid_3[i])
            );
        end
    endgenerate
    assign max_valid_3all = &max_valid_3;
    //第三轮最大值寻找
    wire [15:0]  max_4 [3:0];
    wire [3:0] max_valid_4;
    wire max_valid_4all;
    generate
        for (i = 0; i < 4; i = i + 1) begin : find_max_4
            acc_comparison u_acc_comparison_4 (
                .clk(clk),
                .rst(rst),
                .a(max_3[i]),
                .b(max_3[7 - i]),
                .data_valid(max_valid_3all),
                .max_out(max_4[i]),
                .max_valid(max_valid_4[i])
            );
        end
    endgenerate
    assign max_valid_4all = &max_valid_4;
    //第四轮最大值寻找
    wire [15:0]  max_5 [1:0];
    wire [1:0] max_valid_5;
    wire max_valid_5all;
    generate
        for (i = 0; i < 2; i = i + 1) begin : find_max_5
            acc_comparison u_acc_comparison_5 ( 
                .clk(clk),
                .rst(rst),
                .a(max_4[i]),
                .b(max_4[3 - i]),
                .data_valid(max_valid_4all),
                .max_out(max_5[i]),
                .max_valid(max_valid_5[i])
            );
        end
    endgenerate
    assign max_valid_5all = &max_valid_5;
    //第五轮最大值寻找
    wire max_valid_6;
    wire [15:0] max_out_6;
    acc_comparison u_acc_comparison_6 (
        .clk(clk),
        .rst(rst),
        .a(max_5[0]),
        .b(max_5[1]),
        .data_valid(max_valid_5all),
        .max_out(max_out_6),
        .max_valid(max_valid_6)
    );
    //****************************************
    // 总的最大值比较（跨24轮）
    // 收集每轮64点最大值，共24个，存入max_record中（16位对齐）
    // 再进行分级比较：24->12->6->3->最终
    //****************************************
    reg  [383:0] max_record;   // 24 x 16b = 384b
    reg  [4:0]   cnt;          // 0..23 计数
    reg          last_sample;  // 标记“第24个样本已到达”的打一拍触发
    reg          reduce_start; // 发起最终归约的data_valid脉冲（在完成采样后一拍）

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cnt          <= 5'd0;
            last_sample  <= 1'b0;
            reduce_start <= 1'b0;
        end else begin
            // 默认不触发归约
            reduce_start <= 1'b0;
            last_sample  <= 1'b0;

            if (max_valid_6) begin
                // 写入当前轮的最大值（16位对齐）
                max_record[cnt*16 +: 16] <= max_out_6;
                if (cnt == 5'd23) begin
                    cnt         <= 5'd0;     // 收满12个，回到0
                    last_sample <= 1'b1;     // 下一拍触发归约
                    reduce_start<= 1'b1;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end
        end
    end

    // ---------------------------------
    // 最后比较
    // ---------------------------------

    //第一层：24 -> 12
    wire [15:0] final_s0  [11:0];
    wire [11:0] s0_valid_v;
    wire        s0_valid_all;
    generate
        for (i = 0; i < 12; i = i + 1) begin : find_max_final_0
            acc_comparison u_acc_comparison_final_0 (
                .clk(clk),
                .rst(rst),
                .a(max_record[i*16 +: 16]),
                .b(max_record[(23 - i)*16 +: 16]),
                .data_valid(reduce_start),
                .max_out(final_s0[i]),
                .max_valid(s0_valid_v[i])
            );
        end
    endgenerate
    assign s0_valid_all = &s0_valid_v;

    // 第二层：12 -> 6
    wire [15:0] final_s1  [5:0];
    wire [5:0]  s1_valid_v;
    wire        s1_valid_all;
    generate
        for (i = 0; i < 6; i = i + 1) begin : find_max_final_1
            acc_comparison u_acc_comparison_final_1 (
                .clk(clk),
                .rst(rst),
                .a(final_s0[i]),
                .b(final_s0[11 - i]),
                .data_valid(s0_valid_all),
                .max_out(final_s1[i]),
                .max_valid(s1_valid_v[i])
            );
        end
    endgenerate
    assign s1_valid_all = &s1_valid_v;

    // 第三层：6 -> 3
    wire [15:0] final_s2  [2:0];
    wire [2:0]  s2_valid_v;
    wire        s2_valid_all;
    generate
        for (i = 0; i < 3; i = i + 1) begin : find_max_final_2
            acc_comparison u_acc_comparison_final_2 (
                .clk(clk),
                .rst(rst),
                .a(final_s1[i]),
                .b(final_s1[5 - i]),
                .data_valid(s1_valid_all),
                .max_out(final_s2[i]),
                .max_valid(s2_valid_v[i])
            );
        end
    endgenerate
    assign s2_valid_all = &s2_valid_v;
    
    // --- 最终最大值比较（按 IEEE-754 bf16 数值顺序）---
    // 使用位级排序键：负数取反，非负翻转符号位；
    // 无符号比较这些 key，选择对应原始值。
    wire [15:0] f0 = final_s2[0];
    wire [15:0] f1 = final_s2[1];
    wire [15:0] f2 = final_s2[2];
    wire [15:0] f0_key = f0[15] ? ~f0 : (f0 ^ 16'h8000);
    wire [15:0] f1_key = f1[15] ? ~f1 : (f1 ^ 16'h8000);
    wire [15:0] f2_key = f2[15] ? ~f2 : (f2 ^ 16'h8000);
    wire [15:0] m01     = (f0_key > f1_key) ? f0     : f1;
    wire [15:0] m01_key = (f0_key > f1_key) ? f0_key : f1_key;
    wire [15:0] m012    = (m01_key > f2_key) ? m01    : f2;

    //reg cnt_max;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            max_out   <= 16'b0;
            max_valid <= 1'b0;
            //cnt_max   <= 1'b0;
        end else if (s2_valid_all) begin
            max_out   <= m012;
            max_valid <= 1'b1;
        end else begin
            max_valid <= 1'b0;
        end
    end
endmodule
