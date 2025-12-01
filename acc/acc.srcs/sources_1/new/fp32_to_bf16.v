// Parallel conversion: 32 x FP32 -> 32 x BF16
// Synchronous, single-stage buffering; only valid signaling (downstream always ready)
module fp32_to_bf16 (
    input  wire               clk,
    input  wire               rst,          // sync reset, active-high

    // Upstream handshake
    input  wire [1023:0]      data_in,      // 32 x FP32
    input  wire               data_valid,   // input valid

    // Downstream handshake
    output wire [511:0]      data_out,     // 32 x BF16
    output wire               data_out_valid
);

    // Single-lane converter as a function (round-to-nearest-even, FTZ for subnormal)
    function [15:0] conv32_to_bf16;
        input [31:0] x;
        reg sign;
        reg [7:0] exp;
        reg [22:0] frac;
        reg guard, roundb, sticky;
        reg [6:0] frac_hi7;
        begin
            sign     = x[31];
            exp      = x[30:23];
            frac     = x[22:0];
            guard    = frac[15];
            roundb   = frac[14];
            sticky   = |frac[13:0];
            frac_hi7 = frac[22:16];

            // NaN handling
            if (exp == 8'hFF && frac != 0) begin
                // If upper 7 bits would be zero, force MSB=1 to keep NaN payload non-zero
                if (frac_hi7 == 7'b0) begin
                    conv32_to_bf16 = {sign, 8'hFF, 7'b1000000};
                end else begin
                    // sNaN if frac[22]==0: set MSB to 1; qNaN keep as-is
                    if (frac[22] == 1'b0)
                        conv32_to_bf16 = {sign, 8'hFF, (frac_hi7 | 7'b1000000)};
                    else
                        conv32_to_bf16 = {sign, 8'hFF, frac_hi7};
                end
            end else if (exp == 8'hFF && frac == 0) begin
                // Inf
                conv32_to_bf16 = {sign, 8'hFF, 7'b0};
            end else if (exp == 8'h00) begin
                // Zero or subnormal -> FTZ
                conv32_to_bf16 = {sign, 8'b0, 7'b0};
            end else begin
                // Normal number with RNE rounding using GRS (guard/round/sticky) and tie-even via frac[16]
                if (guard == 1'b0) begin
                    conv32_to_bf16 = {sign, exp, frac_hi7};
                end else begin
                    if (roundb == 1'b1 || sticky == 1'b1 || frac[16] == 1'b1) begin
                        if (frac_hi7 == 7'b1111111) begin
                            // mantissa overflow -> increment exponent
                            if (exp == 8'hFE)
                                conv32_to_bf16 = {sign, 8'hFF, 7'b0};
                            else
                                conv32_to_bf16 = {sign, exp + 8'd1, 7'b0};
                        end else begin
                            conv32_to_bf16 = {sign, exp, frac_hi7 + 7'd1};
                        end
                    end else begin
                        conv32_to_bf16 = {sign, exp, frac_hi7};
                    end
                end
            end
        end
    endfunction

    // 32-lane parallel mapping (combinational conversion of current data_in)
    wire [511:0] conv_bus;
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : g_conv
            wire [15:0] conv_w;
            assign conv_w = conv32_to_bf16(data_in[i*32 +: 32]);
            assign conv_bus[i*16 +: 16] = conv_w;
        end
    endgenerate

    // Single-stage output buffer with valid only
    reg  [511:0] out_data_reg;
    reg           out_valid_reg;

    assign data_out        = out_data_reg;
    assign data_out_valid  = out_valid_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_valid_reg <= 1'b0;
        end else begin
            // 1-cycle latency: valid follows input valid with 1-cycle delay
            out_valid_reg <= data_valid;
            if (data_valid) begin
                // Latch converted result when input valid
                out_data_reg <= conv_bus;
            end
        end
    end

endmodule