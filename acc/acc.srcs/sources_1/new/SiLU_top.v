`timescale 1ns / 1ps


module SiLU_top(
    input  wire        clk,
    input  wire        rst,
    // output reg [511:0]  silu_out,
    // output reg         silu_valid,
    //output wire        ready_in,

    //bf16_2_fp32
    input  wire [1023:0] fp32_data_in,
    input  wire          fp32_valid_in,

    //exp
    output wire [1023:0] inv_out,
    output wire          inv_valid

    //add
    // input  wire [1023:0] add_out,
    // input  wire          add_valid,

    //div
    // output reg  [1023:0] add_out_reg,
    // output reg           div_valid_in_final,
    // output wire [1023:0] fp32_data

    //fp32_to_bf16
    // input  wire [511:0]  silu_out_w,
    // input  wire          silu_valid_w
);

    // =================================
    // 信号定义
    // =================================



    // reg  [4:0]    cnt_in;
    // reg  [4:0]    cnt_div;

    // wire          div_valid_in;


    //assign ready_in = div_valid_in ? 1'b1 : 1'b0;

    inv u_inv (
        .clk(clk),
        .rst(rst),
        .data_in(fp32_data_in),
        .data_valid(fp32_valid_in),
        .data_out(inv_out),
        .data_valid_out(inv_valid)
    );


    // always @(posedge clk) begin
    //     add_out_reg <= add_out;
    // end

    // assign div_valid_in = add_valid ? 1'b1 : 1'b0;
    /*
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            div_valid_in <= 1'b0;
        end else if (add_valid || cnt_div) begin
            div_valid_in <= 1'b1;
        end else begin
            div_valid_in <= 1'b0;
        end
    end
    */
    // blk_mem_gen_0 u_bram_silu (
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

    // always @(posedge clk) begin
    //     div_valid_in_final <= div_valid_in;
    // end 



    // always @(posedge clk or posedge rst) begin
    //     if (rst) begin
    //         silu_valid <= 1'b0;
    //     end else if (silu_valid_w) begin
    //         silu_valid <= 1'b1;
    //         silu_out   <= silu_out_w[511:0];
    //     end else begin
    //         silu_valid <= 1'b0;
    //         silu_out   <= silu_out;
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
    //         cnt_div <= cnt_div;
    //     end
    // end
endmodule
