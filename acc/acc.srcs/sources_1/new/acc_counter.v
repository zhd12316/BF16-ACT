`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/11/20 16:22:32
// Design Name: 
// Module Name: acc_counter
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module acc_counter(
    input wire        clk,
    input wire        rst,
    input wire        start,
    input wire        done,
    output reg [31:0] count
);

    reg counting;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            count    <= 32'd0;
            counting <= 1'b0;
        end else begin
            if (start) begin
                count    <= 32'd0;
                counting <= 1'b1;
            end else if (counting) begin
                count <= count + 32'd1;
                if (done) begin
                    counting <= 1'b0;
                end
            end else begin
                count <= count;
                counting <= counting;
            end
        end
    end

endmodule
