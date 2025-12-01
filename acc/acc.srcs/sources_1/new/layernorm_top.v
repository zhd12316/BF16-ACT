`timescale 1ns / 1ps

module layernorm_top(
    input  wire        clk,
    input  wire        rst,
    //input  wire [511:0]  data_in,
    //input  wire        data_valid, 
    // output reg [511:0]  out_data,
    // output reg        out_valid,

    //bf16_2_fp32
    // input wire [1023:0] fp32_data_in,
    // input wire          fp32_valid_in,

    //sum
    input  wire [31:0]   sum_out,
    input  wire          sum_out_valid,

    //sub
    // output wire [1023:0] fp32_data,
    //output wire [31:0]   everage,
    // output reg           sub_valid_in,
    // input wire [1023:0]  sub_out,
    // input wire           sub_valid,

    //mul
    //input wire           mul_valid,

    //sqrt
    output wire [31:0]   add_out,
    output wire          add_out_valid
    // input  wire          sqrt_valid,

    //div
    // output reg           div_valid_in,
    // output reg  [1023:0] div_in1,
    //input  wire          div_valid,

    //fp32_to_bf16
    // input  wire [511:0]  out_data_w,
    // input  wire          out_valid_w,

    // output reg           div_flag,
    // output reg           div_flag_in
);

    // reg  [4:0]    cnt_in;
    // reg  [4:0]    cnt_sub;
    // reg           add_flag;
    // reg           out_valid_w1;

    // reg           div_valid_in_inside;
    //wire [31:0]   div_in1;
    //wire [31:0]   div_in2;
    //wire          div_valid_in1;
    //wire          div_valid_in2;
    //wire [31:0]   div_out1;
    //wire          div_valid1;

    // reg           write_bram_valid;
    // reg [1023:0]  write_bram_data;

    // wire          everage_valid;
    // wire          bram_read_valid;
    //reg  [31:0]   everage_final;

    // wire [31:0]   add_in;
    // wire          add_in_valid;

    // assign bram_read_valid = (sum_out_valid && ~add_flag) || cnt_sub || sqrt_valid;
    // assign div_in1 = flag ? 32'b0 : sum_out;
    // assign div_valid_in1 = flag ? 1'b0 : sum_out_valid;
    // assign div_in2 = flag ? sum_out : 32'b0;
    // assign div_valid_in2 = flag ? sum_out_valid : 1'b0;
    // assign add_in = sum_out;
    // assign add_in_valid = sum_out_valid && add_flag;
    // assign write_bram_valid = fp32_valid_in || sub_valid;
    // assign write_bram_data = fp32_valid_in ? fp32_data_in : sub_out;

    // blk_mem_gen_0 u_bram_layernorm (
    //     .clka(clk),
    //     .wea(write_bram_valid),
    //     .ena(write_bram_valid),
    //     .addra(cnt_in),
    //     .dina(write_bram_data),
    //     .clkb(clk),
    //     .enb(bram_read_valid),
    //     .addrb(cnt_sub),
    //     .doutb(fp32_data)
    // );


    // always @(posedge clk) begin
    //     write_bram_valid <= fp32_valid_in || sub_valid;
    //     write_bram_data <= fp32_valid_in ? fp32_data_in : sub_out;
    // end

    // floating_point_3 u_fp32_div0 (
    //     .aclk(clk),
    //     .s_axis_a_tvalid(div_valid_in1),
    //     .s_axis_a_tdata(div_in1),
    //     .s_axis_b_tvalid(div_valid_in1),
    //     .s_axis_b_tdata(32'h44400000),
    //     .m_axis_result_tvalid(everage_valid),
    //     .m_axis_result_tdata(everage)
    // );

    // always @(posedge clk) begin
    //     sub_valid_in <= bram_read_valid && ~div_flag_in;
    // end

    // floating_point_3 u_fp32_div1 (
    //     .aclk(clk),
    //     .s_axis_a_tvalid(div_valid_in2),
    //     .s_axis_a_tdata(div_in2),
    //     .s_axis_b_tvalid(div_valid_in2),
    //     .s_axis_b_tdata(32'h44400000),
    //     .m_axis_result_tvalid(div_valid1),
    //     .m_axis_result_tdata(div_out1)
    // );

    floating_point_0 u_fp32_add (
        .aclk(clk),
        .s_axis_a_tvalid(sum_out_valid),
        .s_axis_a_tdata(sum_out),
        .s_axis_b_tvalid(sum_out_valid),
        .s_axis_b_tdata(32'h358637bd), 
        .m_axis_result_tvalid(add_out_valid),
        .m_axis_result_tdata(add_out) // unused
    );

    // always @(posedge clk) begin
    //     div_valid_in_inside      <= bram_read_valid && div_flag_in;
    //     div_valid_in             <= div_valid_in_inside;
    //     div_in1                  <= div_valid_in_inside ? fp32_data : 32'b0;
    // end

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         out_valid <= 1'b0;
    //     end else if (out_valid_w) begin
    //         out_valid <= 1'b1;
    //         out_data   <= out_data_w;
    //     end else begin
    //         out_valid <= 1'b0;
    //         out_data   <= out_data;
    //     end
    // end


    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         cnt_in <= 5'd0;
    //     end else if (write_bram_valid) begin
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

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         cnt_sub <= 5'd0;
    //     end else if (bram_read_valid) begin         
    //         if (cnt_sub == 5'd23) begin
    //             cnt_sub <= 5'd0;
    //         end
    //         else begin
    //             cnt_sub <= cnt_sub + 1'b1;
    //         end
    //     end
    //     else begin
    //         cnt_sub <= 5'd0;
    //     end
    // end

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         out_valid_w1 <= 1'b0;
    //     end else begin
    //         out_valid_w1 <= out_valid_w;
    //     end
    // end

    // always @(posedge clk or posedge rst) begin
    //     if(rst)begin
    //         div_flag <= 1'b1;
    //     end
    //     else if(add_out_valid || (~out_valid_w1 && out_valid_w))begin
    //         div_flag <= ~div_flag;
    //     end
    //     else begin
    //         div_flag <= div_flag;
    //     end
    // end

    // always @(posedge clk or posedge rst) begin
    //     if(rst)begin
    //         add_flag <= 1'b0;
    //     end
    //     else if(sub_valid)begin
    //         add_flag <= 1'b1;
    //     end
    //     else if(fp32_valid_in)begin
    //         add_flag <= 1'b0;
    //     end
    //     else begin
    //         add_flag <= add_flag;
    //     end
    // end

    // always @(posedge clk or posedge rst) begin
    //     if(rst)begin
    //         div_flag_in <= 1'b0;
    //     end
    //     else if(add_out_valid)begin
    //         div_flag_in <= 1'b1;
    //     end
    //     else if(fp32_valid_in)begin
    //         div_flag_in <= 1'b0;
    //     end
    //     else begin
    //         div_flag_in <= div_flag_in;
    //     end
    // end

endmodule
