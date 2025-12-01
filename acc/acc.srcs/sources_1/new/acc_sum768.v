`timescale 1ns / 1ps

// Sum 12 rounds x 64 FP32 = 768 FP32 numbers, processed as 32-lane halves
// - Each input "round" provides 32xFP32 on a[1023:0] with data_valid asserted
// - Per valid, we compute a 32->1 sum (one half). Over 12 rounds x 2 halves = 24 partial sums
// - Then reduce 24 partial sums via 24->12->6->3->2->1 to produce the total over 768 elements

module acc_sum768 (
    input  wire        clk,
    input  wire        rst,
    input  wire [1023:0] sum_in,           // 32 x FP32 per round
    input  wire        data_valid,    // round valid
    output reg  [31:0] sum_total,     // total sum of 12 rounds (FP32)
    output reg         sum_total_valid
);

    // ---------------------------------
    // Per-half sum (32 -> 1)
    // ---------------------------------
    wire [31:0] round_sum;
    wire        round_valid;

    add_sigma u_round_sum (
        .clk       (clk),
        .rst       (rst),
        .data_in   (sum_in),
        .data_valid(data_valid),
        .sum_out   (round_sum),
        .sum_valid (round_valid)
    );

    // ---------------------------------
    // Collect 24 half sums (FP32)
    // ---------------------------------
    reg  [31:0] sum_record [0:23];  // 24 x 32b = 768b
    reg  [4:0]   cnt;         // 0..23 number of collected valid sums
    reg          last_sample; // pulse when 12th sample stored
    reg          reduce_start;// one-cycle pulse to start final reduction

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cnt          <= 5'd0;
            last_sample  <= 1'b0;
            reduce_start <= 1'b0;
        end else begin
            reduce_start <= 1'b0;
            last_sample  <= 1'b0;
            if (round_valid) begin
                sum_record[cnt] <= round_sum;
                if (cnt == 5'd23) begin
                    cnt         <= 5'd0;
                    last_sample <= 1'b1;   // next cycle start reduction
                    reduce_start<= 1'b1;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end
        end
    end

    // ---------------------------------
    // Final reduction of 24 FP32 values: 24->12->6->3->(2->1)
    // ---------------------------------
    // Stage 1: 24 -> 12
    wire [31:0] s1_out [0:11];
    wire [11:0] s1_v;
    wire        s1_all = &s1_v;
    genvar i;
    generate
        for (i = 0; i < 12; i = i + 1) begin : add_s1
            floating_point_0 u_add_s1 (
                .aclk                 (clk),
                .s_axis_a_tvalid      (reduce_start),
                .s_axis_a_tdata       (sum_record[i]),
                .s_axis_b_tvalid      (reduce_start),
                .s_axis_b_tdata       (sum_record[23 - i]),
                .m_axis_result_tvalid (s1_v[i]),
                .m_axis_result_tdata  (s1_out[i])
            );
        end
    endgenerate

    // Stage 2: 12 -> 6
    wire [31:0] s2_out [0:5];
    wire [5:0]  s2_v;
    wire        s2_all = &s2_v;
    generate
        for (i = 0; i < 6; i = i + 1) begin : add_s2
            floating_point_0 u_add_s2 (
                .aclk                 (clk),
                .s_axis_a_tvalid      (s1_all),
                .s_axis_a_tdata       (s1_out[i]),
                .s_axis_b_tvalid      (s1_all),
                .s_axis_b_tdata       (s1_out[11 - i]),
                .m_axis_result_tvalid (s2_v[i]),
                .m_axis_result_tdata  (s2_out[i])
            );
        end
    endgenerate

    // Stage 3: 6 -> 3
    wire [31:0] s3_out [0:2];
    wire [2:0]  s3_v;
    wire        s3_all = &s3_v;
    generate
        for (i = 0; i < 3; i = i + 1) begin : add_s3
            floating_point_0 u_add_s3 (
              .aclk                 (clk),
              .s_axis_a_tvalid      (s2_all),
              .s_axis_a_tdata       (s2_out[i]),
              .s_axis_b_tvalid      (s2_all),
              .s_axis_b_tdata       (s2_out[5 - i]),
              .m_axis_result_tvalid (s3_v[i]),
              .m_axis_result_tdata  (s3_out[i])
            );
        end
    endgenerate

    // Stage 4: 3 -> 2 (pair first two)
    reg  [31:0] s3_out_reg;
    wire [31:0] s4_pair;
    wire        s4_v;
    floating_point_0 u_add_s4_pair (
        .aclk                 (clk),
        .s_axis_a_tvalid      (s3_all),
        .s_axis_a_tdata       (s3_out[0]),
        .s_axis_b_tvalid      (s3_all),
        .s_axis_b_tdata       (s3_out[1]),
        .m_axis_result_tvalid (s4_v),
        .m_axis_result_tdata  (s4_pair)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            s3_out_reg <= 32'b0;
        end else if (s3_all) begin
            s3_out_reg <= s3_out[2];
        end
        else begin
            s3_out_reg <= s3_out_reg;
        end
    end

    // Stage 5: 2 -> 1 (add the third)
    wire [31:0] s5_final;
    wire        s5_v;
    floating_point_0 u_add_s5_final (
        .aclk                 (clk),
        .s_axis_a_tvalid      (s4_v),
        .s_axis_a_tdata       (s4_pair),
        .s_axis_b_tvalid      (s4_v),
        .s_axis_b_tdata       (s3_out_reg),
        .m_axis_result_tvalid (s5_v),
        .m_axis_result_tdata  (s5_final)
    );

    // Output register
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sum_total       <= 32'b0;
            sum_total_valid <= 1'b0;
        end else if (s5_v) begin
            sum_total       <= s5_final;
            sum_total_valid <= 1'b1;
        end else begin
            sum_total_valid <= 1'b0;
            sum_total       <= sum_total;
        end
    end

endmodule
