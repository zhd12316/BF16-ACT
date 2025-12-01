`timescale 1ns/1ps

// Testbench: load all X/Y bf16 (512-bit per beat) first, then pulse start
// Output: consume 1024-bit (64 bf16) on out_data_el when out_valid=1; ignore out_data

module tb_el_mul_load_start;
  // Clock/Reset
  reg clk = 0;
  reg rst = 1;
  always #5 clk = ~clk; // 100 MHz

  // DUT I/O
  reg  [511:0] data_in;
  reg          data_valid;
  reg          start;
  reg  [2:0]   mode;
  wire [511:0] out_data;       // unused per new requirement
  wire [1023:0] out_data_el;   // 64 bf16
  wire         out_valid;
  // DUT done/count outputs (capture for latency/throughput measurement)
  wire         done;
  wire [31:0]  count;

  // Parameters
  localparam integer ELEMS_PER_ROW = 768;     // elements per row (bf16)
  localparam integer BLK_ELEMS     = 32;      // 32 bf16 => 512-bit per transfer
  localparam integer ROW_BLOCKS    = ELEMS_PER_ROW / BLK_ELEMS; // 24
  localparam [2:0]   MODE_EL_MUL   = 3'b101;  // final_top mode for element-wise mul
  // Total rows to preload before start
  localparam integer NUM_ROWS      = 64;
  // inactivity watchdog (cycles) to avoid hanging if DUT stalls
  localparam integer IDLE_LIMIT   = 200000;

  // File paths (adjust if needed)
  // Note: Verilog 2001 has no 'string' type; paths used directly in $fopen literals.

  // Simple memories to store one row of X/Y (bf16 = 16b)
  reg [15:0] mem_x [0:ELEMS_PER_ROW-1];
  reg [15:0] mem_y [0:ELEMS_PER_ROW-1];

  // -----------------------------------------
  // DUT instance: final_top
  // -----------------------------------------
  final_top dut (
    .clk       (clk),
    .rst       (rst),
    .data_in   (data_in),
    .data_valid(data_valid),
    .start     (start),
    .mode      (mode),
    .out_data  (out_data),     // ignored
    .out_data_el(out_data_el),
    .out_valid (out_valid),
    .done      (done),
    .count     (count)
  );

  // -----------------------------------------
  // Helpers
  // -----------------------------------------
  task drive_idle;
    begin
      data_in    <= {512{1'b0}};
      data_valid <= 1'b0;
      start      <= 1'b0;
    end
  endtask

  // Pack 32 bf16 words into 512-bit bus and send for 1 cycle
  task send_block_vec(input [511:0] payload);
    begin
      @(posedge clk);
      data_in    <= payload;
      data_valid <= 1'b1;
      @(posedge clk);
      data_valid <= 1'b0;
    end
  endtask

  // Send one row continuously: data_valid keeps high, data_in updates every cycle
  // Interleave blocks as X0, Y0, X1, Y1, ... to match BRAM packing
  task send_row_xy(input integer base_idx);
    integer idx, blk, i;
    reg [511:0] payload;
    begin
      // Stream 2*ROW_BLOCKS cycles without gaps
      for (idx = 0; idx < (ROW_BLOCKS<<1); idx = idx + 1) begin
        blk = idx >> 1; // block index 0..ROW_BLOCKS-1
        payload = {512{1'b0}};
        if (idx[0] == 1'b0) begin
          // even idx: X block
          for (i = 0; i < BLK_ELEMS; i = i + 1)
            payload[i*16 +: 16] = mem_x[base_idx + blk*BLK_ELEMS + i];
        end else begin
          // odd idx: Y block
          for (i = 0; i < BLK_ELEMS; i = i + 1)
            payload[i*16 +: 16] = mem_y[base_idx + blk*BLK_ELEMS + i];
        end
        @(posedge clk);
        data_in    <= payload;
        data_valid <= 1'b1;
      end
      // Deassert after last beat
      @(posedge clk);
      data_valid <= 1'b0;
    end
  endtask

  // Write 1024-bit out_data_el as 64 bf16 (little-endian bytes) to binary file
  task write_out1024_bf16(input integer fd, input [1023:0] data1024);
    integer i;
    reg [15:0] w;
    reg [7:0] b0, b1;
    begin
      for (i = 0; i < 64; i = i + 1) begin
        w  = data1024[i*16 +: 16];
        b0 = w[7:0];
        b1 = w[15:8];
        // write 2 bytes per bf16 (little-endian)
        $fwrite(fd, "%c", b0);
        $fwrite(fd, "%c", b1);
      end
    end
  endtask

  // -----------------------------------------
  // Stimulus
  // -----------------------------------------
  integer fx, fy, fo;
  integer i;
  // extra temporaries for byte reads
  reg [7:0] lo_byte;
  reg [7:0] hi_byte;
  integer beats_expected;
  integer beats_got;
  // DUT count capture
  integer measured_count; // saved DUT-provided count (start->done cycles)
  integer done_cycle;     // cycle when done asserted (from DUT)
  reg     prev_done;      // previous done value for edge detection
  // loop temps for streaming all rows
  integer row, idx, blk, j;
  reg [511:0] payload;
  // simulation cycle counter and start/end capture
  reg [63:0] sim_cycle;
  reg [63:0] start_cycle;
  reg [63:0] end_cycle;
  reg        start_seen;
  reg [63:0] cycles_elapsed;

  // increment simulation cycle count on every clock
  always @(posedge clk) begin
    if (rst) begin
      sim_cycle <= 0;
      start_seen <= 0;
      start_cycle <= 0;
      end_cycle   <= 0;
      // init DUT-done capture helpers
      measured_count <= -1;
      done_cycle <= 0;
      prev_done <= 1'b0;
    end else begin
      sim_cycle <= sim_cycle + 1;
      // capture start rising edge globally so fork timing does not matter
      if (start && !start_seen) begin
        start_seen  <= 1'b1;
        start_cycle <= sim_cycle; // cycle when start asserted
      end
      // capture DUT done rising edge and read count once
      if (done && !prev_done && measured_count == -1) begin
        measured_count <= count;
        done_cycle <= sim_cycle;
        $display("[DBG] DUT asserted done at sim_cycle=%0d, count=%0d", sim_cycle, count);
      end
      prev_done <= done;
    end
  end

  initial begin
    drive_idle();
    mode <= MODE_EL_MUL;
    sim_cycle   = 0;
    start_cycle = 0;
    end_cycle   = 0;
    start_seen  = 0;

    // Reset
    repeat (10) @(posedge clk);
    rst <= 0;

    // Load one row (768 elems) for X/Y from binary files (uint16 little-endian)
    // Note: paths use forward slashes for simulator compatibility
    fx = $fopen("C:/Users/ASUS/Desktop/acc/data/X_test_tensor_bf16.bin", "rb");
    if (fx == 0) begin
      $display("[FATAL] Cannot open X file");
      $finish;
    end
    fy = $fopen("C:/Users/ASUS/Desktop/acc/data/Y_test_tensor_bf16.bin", "rb");
    if (fy == 0) begin
      $display("[FATAL] Cannot open Y file");
      $finish;
    end

    // (Defer reading and closing; we will stream all rows below)

      // $display("[TB] Start streaming %0d rows continuously...", NUM_ROWS);

    // Stream ALL rows continuously: keep data_valid high throughout
    // Prime
    data_valid <= 1'b0;
    for (row = 0; row < NUM_ROWS; row = row + 1) begin
      // Load a row from files
      for (i = 0; i < ELEMS_PER_ROW; i = i + 1) begin
        if ($fread(lo_byte, fx) != 1) begin $display("[FATAL] X underflow at row %0d, idx %0d", row, i); $finish; end
        if ($fread(hi_byte, fx) != 1) begin $display("[FATAL] X underflow at row %0d, idx %0d", row, i); $finish; end
        mem_x[i] = {hi_byte, lo_byte};
      end
      for (i = 0; i < ELEMS_PER_ROW; i = i + 1) begin
        if ($fread(lo_byte, fy) != 1) begin $display("[FATAL] Y underflow at row %0d, idx %0d", row, i); $finish; end
        if ($fread(hi_byte, fy) != 1) begin $display("[FATAL] Y underflow at row %0d, idx %0d", row, i); $finish; end
        mem_y[i] = {hi_byte, lo_byte};
      end
      // Stream this row blocks: X0,Y0,X1,Y1,... with data_valid high
      for (idx = 0; idx < (ROW_BLOCKS<<1); idx = idx + 1) begin
        blk = idx >> 1;
        payload = {512{1'b0}};
        if (idx[0] == 1'b0) begin
          for (j = 0; j < BLK_ELEMS; j = j + 1)
            payload[j*16 +: 16] = mem_x[blk*BLK_ELEMS + j];
        end else begin
          for (j = 0; j < BLK_ELEMS; j = j + 1)
            payload[j*16 +: 16] = mem_y[blk*BLK_ELEMS + j];
        end
        @(posedge clk);
        data_in    <= payload;
        data_valid <= 1'b1;
      end
      // Do NOT deassert data_valid here to keep continuous stream across rows
    end
    // Now drop data_valid one cycle after the last beat
    @(posedge clk);
    data_valid <= 1'b0;

    $display("[TB] Completed streaming all %0d rows.", NUM_ROWS);
    // Close input files now that streaming is complete
    $fclose(fx);
    $fclose(fy);

    // Wait a few cycles for bf16->fp32 pipelines to flush into BRAM, then start
    repeat (16) @(posedge clk);
    start <= 1'b1; @(posedge clk); start <= 1'b0;

    // Capture outputs
    fo = $fopen("C:/Users/ASUS/Desktop/acc/data/eltwise_mul_out_tb.bin", "wb");
    if (fo == 0) begin
      $display("[FATAL] Cannot open output file");
      $finish;
    end

    // Expect ELEMS_PER_ROW/64 beats of 1024-bit output per row pair => 768/64 = 12
    beats_expected = (ELEMS_PER_ROW / 64) * NUM_ROWS;
    beats_got = 0;
    $display("[TB] Expect ~%0d 1024-bit beats", beats_expected);

    // Wait for out_valid and dump
    fork
      begin : timeout_block
        // Simple timeout: 2 ms sim time
        #(2_000_000);
        $display("[FATAL] Timeout waiting for outputs. Got=%0d", beats_got);
        $finish;
      end
      begin : consume_block
        integer idle;
        idle = 0;
        forever begin
          @(posedge clk);
          if (out_valid) begin
            write_out1024_bf16(fo, out_data_el);
            beats_got = beats_got + 1;
            idle = 0;
            // if this was the last expected beat, capture end cycle
            if (beats_got >= beats_expected) begin
              end_cycle <= sim_cycle;
            end
          end else begin
            idle = idle + 1;
          end
          if (beats_got >= beats_expected) begin
            disable timeout_block;
            disable consume_block;
          end
          if (idle > IDLE_LIMIT) begin
            $display("[WARN] Output idle > %0d cycles; stopping early. Got=%0d / %0d", IDLE_LIMIT, beats_got, beats_expected);
            end_cycle <= sim_cycle;
            disable timeout_block;
            disable consume_block;
          end
        end
      end
    join

    $fclose(fo);
    $display("[TB] Done. Wrote %0d beats to eltwise_add_out_tb.bin", beats_got);
    // Wait a short window for DUT 'done' to assert (it may come slightly after
    // the last output beat). This prevents false negatives where we print
    // "DUT did not assert done" even though done arrives a few cycles later.
    begin : wait_for_done_block
      integer wait_cnt;
      wait_cnt = 0;
      // wait up to 2000 cycles (adjust if your DUT needs more)
      while (measured_count == -1 && wait_cnt < 2000) begin
        @(posedge clk);
        wait_cnt = wait_cnt + 1;
      end
      if (measured_count != -1) begin
        $display("[DONE] done_cycle=%0d DUT count=%0d (captured after wait %0d cycles)", done_cycle, measured_count, wait_cnt);
      end else begin
        $display("[DONE] DUT did not assert done within %0d cycles after outputs. measured_count=%0d", wait_cnt, measured_count);
      end
      // Debug: print raw DUT done and count at this point
      $display("[DBG] At end-of-wait: DUT done=%b, DUT count=%0d", done, count);
    end
    if (start_seen) begin
      // If end_cycle was not set (==0), use current sim_cycle as end
      if (end_cycle == 64'd0) begin
        end_cycle = sim_cycle;
      end
      // compute elapsed safely avoiding unsigned wrap
      if (end_cycle >= start_cycle) begin
        cycles_elapsed = end_cycle - start_cycle + 1;
      end else begin
        cycles_elapsed = sim_cycle - start_cycle + 1;
      end
      $display("[TB] start_cycle=%0d, end_cycle=%0d, cycles_elapsed=%0d", start_cycle, end_cycle, cycles_elapsed);
    end else begin
      $display("[TB] start was never asserted during sim (start_seen=0)");
    end
    #100;
    $finish;
  end

endmodule
