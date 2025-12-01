module acc_comparison (
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0]  a,
    input  wire [15:0]  b,
    input  wire        data_valid,
    output reg  [15:0]  max_out,
    output reg         max_valid
);
    // Order-preserving key for IEEE-754 bfloat16 comparison using unsigned '>'
    // Mapping rule (standard trick for floats):
    //   key = (sign ? ~bits : (bits ^ 16'h8000))
    // This ensures unsigned key ordering matches numeric ordering, including negatives.
    wire [15:0] a_key = a[15] ? ~a : (a ^ 16'h8000);
    wire [15:0] b_key = b[15] ? ~b : (b ^ 16'h8000);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            max_valid <= 1'b0;
        end else if(data_valid)begin
            max_valid <= 1'b1;
            // Select original operand corresponding to larger key
            max_out <= (a_key > b_key) ? a : b;
        end
        else begin
            max_valid <= 1'b0;
            max_out <= max_out;
        end
    end
endmodule