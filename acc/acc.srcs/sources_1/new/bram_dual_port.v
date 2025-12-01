`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/11/05 10:24:46
// Design Name: 
// Module Name: bram_dual_port
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


module bram_dual_port(
    input clk,

    input wea,                  //enable write signal of channel a
    input ena,
    input enb,                  //enable signal of channel b

    input [10:0] addra,          //address of channel a
    input [10:0] addrb,          //address of channle b

    input [1023:0] dina,      //data input of channel a
    output reg [1023:0] doutb //data output of channel b
);
(*ram_style = "reg" *)reg [1023:0] RAM [1535:0];         //DATAWIDTH = 1024, DEPTH = 1536

always @(posedge clk) begin     //write channel
    if(wea && ena) begin
        RAM[addra] <= dina;
    end
end

always @(posedge clk) begin    //read channel
    if(enb) begin
        doutb <= RAM[addrb];
    end
end
endmodule
