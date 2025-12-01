`timescale 1ns / 1ps

module RMSnorm_top(
    input  wire        clk,
    input  wire        rst,
    // output reg [511:0] out_data,
    // output reg         out_valid,
    //output wire        ready_in,
    //input  wire [511:0] data_in,
    //input  wire         data_valid,

    //bf16_2_fp32
    // input  wire [1023:0] fp32_data_in,
    // input  wire          fp32_valid_in,

    //mul
    //input  wire [1023:0] mul_out, 
    //input  wire          mul_valid,
    // output   reg           flag_div,

    //sum
    input  wire [31:0]   sum_out,
    input  wire          sum_valid,

    //sqrt
    output wire [31:0]   add_out,
    output wire          add_out_valid
    // input  wire [31:0]   sqrt_out,
    // input  wire          sqrt_valid,
    // output reg           div_valid_in_final

    //div
    // output wire [1023:0] fp32_data,
    // output reg           div_valid_in_final

    //fp32_to_bf16
    // input  wire [511:0]  out_data_w,
    // input  wire          out_valid_w
);


    // reg  [4:0]    cnt_in;
    // reg  [4:0]    cnt_div;


    // assign div_valid_in = sqrt_valid || cnt_div;


    // blk_mem_gen_0 u_bram_layernorm (
    //     .clka(clk),
    //     .wea(fp32_valid_in),
    //     .ena(fp32_valid_in),
    //     .addra(cnt_in),
    //     .dina(fp32_data_in),
    //     .clkb(clk),
    //     .enb(div_valid_in),
    //     .addrb(cnt_div),
    //     .doutb(fp32_data)
    // );


    floating_point_0 u_fp32_add (
        .aclk(clk),
        .s_axis_a_tvalid(sum_valid),
        .s_axis_a_tdata(sum_out),
        .s_axis_b_tvalid(sum_valid),
        .s_axis_b_tdata(32'h358637bd), 
        .m_axis_result_tvalid(add_out_valid),
        .m_axis_result_tdata(add_out) 
    );


    // always @(posedge clk) begin
    //     div_valid_in_final <= div_valid_in;
    // end

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         out_valid <= 1'b0;
    //     end else if (out_valid_w) begin
    //         out_valid <= 1'b1;
    //         out_data  <= out_data_w;
    //     end else begin
    //         out_valid <= 1'b0;
    //         out_data  <= out_data;
    //     end
    // end

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

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         cnt_div <= 5'd0;
    //     end else if (div_valid_in) begin         
    //         if (cnt_div == 5'd23) begin
    //             cnt_div <= 5'd0;
    //         end
    //         else begin
    //             cnt_div <= cnt_div + 1'b1;
    //         end
    //     end
    //     else begin
    //         cnt_div <= 5'd0;
    //     end
    // end

    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         flag_div <= 1'b0;
    //     end else if (add_out_valid) begin         
    //         flag_div <= 1'b1;
    //     end
    //     else if(fp32_valid_in)begin
    //         flag_div <= 1'b0;
    //     end
    //     else begin
    //         flag_div <= flag_div;
    //     end
    // end

endmodule
