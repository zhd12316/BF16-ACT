`timescale 1ns / 1ps


module softmax_top(
    input  wire        clk,
    input  wire        rst,
    input  wire [511:0]  softmax_data_in,
    input  wire        data_valid,
    // output reg [511:0]  softmax_out,
    // output reg         softmax_valid,
    // output wire        ready_in,

    //bf16_2_fp32
    // input  wire [1023:0] fp32_data_in,
    // input  wire          fp32_valid_in,

    //max
    output wire           max_valid,
    //sub
    // output reg            sub_valid_in_final,
    output reg  [31:0]    max_val_fp32,
    output wire [1023:0]  fp32_data,

    //exp
    // input  wire [1023:0]  exp_out,
    // input  wire           exp_valid,

    //sum
    input  wire           sum_valid,

    //div
    output reg            div_valid_in_final
    // output wire [1023:0]  fp32_data_exp

    //fp32_to_bf16
    // input  wire [511:0]  softmax_out_w,
    // input  wire         softmax_valid_w
);
    // =================================
    // 信号定义
    // =================================

    wire [15:0] max_val;
    // reg  [15:0] max_val_ram [0:63];
    wire        max_valid_in;
    wire        sub_valid_in;


    reg  [4:0] delay_cnt; //24rounds
    // reg  [4:0] cnt_in;
    // reg  [4:0] cnt_exp;
    reg  [4:0] cnt_sum;
    reg  [5:0] cnt_max;
    // wire [4:0] addra;


    wire        div_valid_in;    // start division when sum is ready (note: requires buffering numerators for correctness)
    // reg         div_valid_in1;

    // assign ready_in = sum_valid ? 1'b1 : 1'b0;

    // ===============================
    //bram读写
    // ===============================

    // bram_dual_port u_blk_mem_gen (
    //     .clk(clk),
    //     .ena(fp32_valid_in),
    //     .wea(fp32_valid_in),
    //     .addra(cnt_in),
    //     .dina(fp32_data_in),
    //     .enb(sub_valid_in),
    //     .addrb(delay_cnt),
    //     .doutb(fp32_data)
    // );

    // bram_dual_port u_blk_mem_gen_exp (
    //     .clk(clk),
    //     .ena(exp_valid),
    //     .wea(exp_valid),
    //     .addra(cnt_exp),
    //     .dina(exp_out),
    //     .enb(div_valid_in1),
    //     .addrb(cnt_div),
    //     .doutb(fp32_data_exp)
    // );

    // =================================
    //接收输入数据，64个BF16扩展为FP32
    // =================================

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         cnt_in <= 5'd0;
    //     end else if (fp32_valid_in) begin           
    //         if (cnt_in == 5'd23) begin
    //             cnt_in <= 5'd0;
    //         end
    //         else begin
    //             cnt_in <= cnt_in + 1'b1;
    //         end
    //     end
    //     else begin
    //         cnt_in <= cnt_in;
    //     end
    // end

    // =================================
    // 最大值计算流程
    // =================================

    acc_findmax u_acc_findmax (
        .clk(clk),
        .rst(rst),
        .data_in(softmax_data_in),
        .data_valid(data_valid),
        .max_out(max_val),
        .max_valid(max_valid)
    );
    // ===============================
    // BF16 最大值扩展为 FP32
    // ===============================
    always @(posedge clk) begin
        // max_val_ram[cnt_max]   <= max_val;
        // max_valid              <= max_valid_in;
        max_val_fp32           <= {max_val,16'b0};
    end


    // =================================
    // 减法计算流程
    // =================================
    assign sub_valid_in = max_valid || delay_cnt ? 1'b1 : 1'b0;

    // always @(posedge clk) begin
    //     sub_valid_in_final <= sub_valid_in;
    // end



    // =================================
    // 除法模块实例化
    // =================================


    assign div_valid_in = sum_valid || cnt_sum ? 1'b1 : 1'b0;

    always @(posedge clk) begin
        // div_valid_in       <= sum_valid || cnt_sum ? 1'b1 : 1'b0;
        // div_valid_in1      <= div_valid_in;
        // cnt_div            <= cnt_sum;
        div_valid_in_final <= div_valid_in;
    end    




    // 输出寄存
    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         softmax_valid <= 1'b0;
    //     end else if (softmax_valid_w) begin
    //         softmax_out <= softmax_out_w[511:0];
    //         softmax_valid <= softmax_valid_w;
    //     end
    //     else begin
    //         softmax_out <= softmax_out;
    //         softmax_valid <= 1'b0;
    //     end
    // end

    // =================================
    // 延时计数器
    // =================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            delay_cnt <= 5'd0;
        end else if (sub_valid_in) begin
            if (delay_cnt == 5'd23) begin
                delay_cnt <= 5'd0;
            end
            else begin
                delay_cnt <= delay_cnt + 1'b1;
            end
        end
        else begin
            delay_cnt <= delay_cnt;
        end
    end

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         cnt_exp <= 5'd0;
    //     end else if (exp_valid) begin
    //         if (cnt_exp == 5'd23) begin
    //             cnt_exp <= 5'd0;
    //         end
    //         else begin
    //             cnt_exp <= cnt_exp + 1'b1;
    //         end
    //     end
    //     else begin
    //         cnt_exp <= cnt_exp;
    //     end
    // end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cnt_sum <= 5'd0;
        end else if (div_valid_in) begin
            if (cnt_sum == 5'd23) begin
                cnt_sum <= 5'd0;
            end
            else begin
                cnt_sum <= cnt_sum + 1'b1;
            end
        end
        else begin
            cnt_sum <= cnt_sum;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cnt_max <= 6'd0;
        end else if (delay_cnt == 5'd23) begin
            if (cnt_max == 6'd63) begin
                cnt_max <= 6'd0;
            end
            else begin
                cnt_max <= cnt_max + 1'b1;
            end
        end
        else begin
            cnt_max <= cnt_max;
        end
    end

endmodule
