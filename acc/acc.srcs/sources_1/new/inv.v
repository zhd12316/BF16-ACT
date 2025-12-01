`timescale 1ns / 1ps



module inv(
    input  wire        clk,
    input  wire        rst,
    input  wire [1023:0]  data_in,
    input  wire        data_valid,
    output reg [1023:0]  data_out,
    output reg         data_valid_out
);
    wire [1023:0] data_in_neg;
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin
            assign data_in_neg[i*32 +: 32] = {
                ~data_in[i*32 + 31],                    // sign
                data_in[i*32 + 30 : i*32 + 0]          // exponent + fraction
            };
        end
    endgenerate

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_valid_out <= 1'b0;
        end else if (data_valid) begin
            data_out <= data_in_neg;
            data_valid_out <= 1'b1;
        end else begin
            data_valid_out <= 1'b0;
        end
    end
endmodule
