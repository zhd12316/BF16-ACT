`timescale 1ns/1ps

module tb_rmsnorm;
	// Clock/Reset
	reg clk;
	reg rst;

	// DUT ports
	reg  [511:0]  data_in;      // 32 x BF16 input bus（每拍 512bit）
	reg           data_valid;   // input valid (one-cycle strobe per round)
	wire [511:0]  softmax_out;  // BF16 output bus（每拍 512bit，32 路 BF16）
	wire          softmax_valid;// output valid（逐轮脉冲，每轮 512bit）
	reg           start;        // 试验启动信号：与 data_valid 同步拉高
	wire          done;         // 结束脉冲，由 DUT 给出
	wire [31:0]   count;        // DUT 计数（从 start 到 done 的周期数）

	// 测试侧就绪信号：仅当 ready_in=1 时才发送一轮输入
	reg           ready_in;
	parameter RMSNORM_MODE = 3'b010; //  gelu模式
	// 实例化 DUT
	final_top dut (
		.clk(clk),
		.rst(rst),
		.start(start),
		.data_in(data_in),
		.data_valid(data_valid),
		.out_data(softmax_out),
		.out_valid(softmax_valid),
		.done(done),
		.count(count),
		.mode(RMSNORM_MODE)
	);

	// Clock generation: 10ns period
	initial clk = 1'b0;
	always #5 clk = ~clk;

	// 配置：当 ready_in=1 时，每拍输入 512bit（32xBF16），每行输入 24 轮；
	//       out_valid=1 时，每拍读出 512bit（32xBF16）。总计 64 行，每行 24 轮。
	localparam integer N_ROWS            = 64;
	localparam integer COLS_PER_ROW      = 768;     // 32 lanes x 24 rounds
	localparam integer ROUNDS_PER_ROW    = 24;
	localparam integer OUTPUT_CHUNK_BITS = 512;     // 每拍输出 512bit，32 路 BF16
	localparam integer CHUNK_LANES       = OUTPUT_CHUNK_BITS/16; // 32 lanes per output
	localparam integer EXPECT_CHUNKS     = N_ROWS*ROUNDS_PER_ROW; // 期望输出总轮数

	// 全量输入缓存：a[row] 含 24 个 512bit chunk（共 768 个 bf16）
	reg [512*ROUNDS_PER_ROW-1:0] a [0:N_ROWS-1];

	// 测试模式（默认 one-hot）：0=全1，1=one-hot（默认1）
	integer g_pattern = 1;
	// one-hot 的全局热点索引（默认 760，可通过 +HOT=nnn 覆盖）
	integer g_hot_idx = 710;

	// BF16 常量
	localparam [15:0] BF16_ONE  = 16'h3F80; // 1.0
	localparam [15:0] BF16_ZERO = 16'h0000; // 0.0
	localparam [15:0] BF16_BIG  = 16'h4100; // 8.0

	// 由 g_hot_idx 推导出的信息（仅用于打印提示，不参与比较）
	integer hot_round;
	integer hot_lane32;
	integer hot_chunk;      // 0..23（与 DUT 输出顺序对齐，每拍一轮）

	task compute_expect;
		begin
			hot_round   = g_hot_idx / 32;
			hot_lane32  = g_hot_idx % 32;
			// 输出按 512b 每拍一轮，等价于输入 round 对齐
			hot_chunk   = hot_round; // 0..23
		end
	endtask

	// 每轮的 BF16 常量（模式2用）：为 r=0..23（按 r%12 循环）选择一组不同的数值，范围约 0.0~1.375
	function [15:0] bf16_round_value;
		input integer r;
		begin
			case (r % 12)
				0:  bf16_round_value = 16'h0000; // 0.0
				1:  bf16_round_value = 16'h3E00; // 0.125
				2:  bf16_round_value = 16'h3E80; // 0.25
				3:  bf16_round_value = 16'h3EC0; // 0.375
				4:  bf16_round_value = 16'h3F00; // 0.5
				5:  bf16_round_value = 16'h3F20; // 0.625
				6:  bf16_round_value = 16'h3F40; // 0.75
				7:  bf16_round_value = 16'h3F60; // 0.875
				8:  bf16_round_value = 16'h3F80; // 1.0
				9:  bf16_round_value = 16'h3F90; // 1.125
				10: bf16_round_value = 16'h3FA0; // 1.25
				11: bf16_round_value = 16'h3FB0; // 1.375
				default: bf16_round_value = 16'h0000;
			endcase
		end
	endfunction

	// 生成一轮 32xBF16 的输入 payload（每拍 512b）
	function [511:0] gen_round_payload;
		input integer round_idx; // 0..23
		integer i;
		integer hr;
		integer hl;
		reg [15:0] v_per_round;
		begin
			if (g_pattern == 0) begin
				// Uniform 1.0
				for (i = 0; i < 32; i = i + 1) begin
					gen_round_payload[i*16 +: 16] = BF16_ONE;
				end
			end else if (g_pattern == 1) begin
				// One-hot across 768: 仅 (hot_round, hot_lane) 位置为 BF16_BIG，其余为 0
				hr = g_hot_idx / 32;
				hl = g_hot_idx % 32;
				for (i = 0; i < 32; i = i + 1) begin
					gen_round_payload[i*16 +: 16] = (round_idx==hr && i==hl) ? BF16_BIG : BF16_ZERO;
				end
			end else if (g_pattern == 2) begin
				// 同一轮内 32 路相同，不同轮取不同常量
				v_per_round = bf16_round_value(round_idx);
				for (i = 0; i < 32; i = i + 1) begin
					gen_round_payload[i*16 +: 16] = v_per_round;
				end
			end else begin
				// 默认退化为全 0
				for (i = 0; i < 32; i = i + 1) begin
					gen_round_payload[i*16 +: 16] = BF16_ZERO;
				end
			end
		end
	endfunction

	// 仅当 ready_in==1 时才发送一轮数据（data_valid 在一个完整时钟内为 1）
	task send_round;
		input [511:0] payload;
		begin
			while (ready_in == 1'b0) @(posedge clk);
			@(posedge clk);
			data_in <= payload;
			data_valid <= 1'b1;
			@(posedge clk);
			data_valid <= 1'b0;
		end
	endtask

	// 可配置的 ready_in 行为
	integer READY_CONST; // -1=未指定；0/1=常值
	integer READY_HIGH;  // >0 时启用周期模式，高电平周期数
	integer READY_LOW;   // >0 时启用周期模式，低电平周期数
	integer ready_cnt;

	initial begin
		ready_in    = 1'b1;  // 默认常高
		READY_CONST = -1;    // -1 表示未指定常值
		READY_HIGH  = 0;     // 0 表示未启用周期模式
		READY_LOW   = 0;
		ready_cnt   = 0;
	end

	always @(posedge clk or posedge rst) begin
		if (rst) begin
			ready_cnt <= 0;
			if (READY_CONST != -1) ready_in <= (READY_CONST != 0);
			else ready_in <= 1'b1;
		end else begin
			if (READY_CONST != -1) begin
				// 常值模式
				ready_in <= (READY_CONST != 0);
			end else if (READY_HIGH > 0 && READY_LOW > 0) begin
				// 周期性抖动：保持 READY_HIGH 个周期为 1，然后 READY_LOW 个周期为 0，循环
				ready_cnt <= ready_cnt + 1;
				if (ready_in) begin
					if (ready_cnt >= READY_HIGH) begin
						ready_in  <= 1'b0;
						ready_cnt <= 0;
					end
				end else begin
					if (ready_cnt >= READY_LOW) begin
						ready_in  <= 1'b1;
						ready_cnt <= 0;
					end
				end
			end else begin
				// 默认常高
				ready_in <= 1'b1;
			end
		end
	end

	// 监控 out_valid 并打印 32 路 BF16（每拍 512b）
	integer cycle;
	integer first_valid_cycle;
	integer last_valid_cycle;
	integer num_outputs;
	// Latency measurement: first input (data_valid) -> first output (softmax_valid)
	integer first_in_cycle;
	integer first_out_cycle;
	integer latency_cycles;
	integer i_print;
	// 输出文件句柄与 16bit 暂存（需在此位置声明，供 always 块使用）
	integer fd_out;
	reg [15:0] word16;

	integer done_cycle;          // 捕获 done = 1 的周期
	integer measured_count;      // 保存 count（DUT 提供的周期数）
	reg     prev_done;           // 用于检测 done 上升沿（处理1周期脉冲）
	integer start_cycle_tb;      // TB 本地记录的 start 周期（第一次 data_valid）
	integer elapsed_cycles_tb;   // TB 计算的 start->done 周期数
	reg data_valid_d; // delayed copy for edge detect
	initial cycle = 0;
	// Generate a single-cycle start pulse when data_valid rises from 0->1
	always @(posedge clk) begin
		if (rst) begin
			start <= 1'b0;
			data_valid_d <= 1'b0;
		end else begin
			start <= data_valid && ~data_valid_d;
			data_valid_d <= data_valid;
		end
	end
	always @(posedge clk) begin
		if (rst) begin
			cycle            <= 0;
			first_valid_cycle <= -1;
			last_valid_cycle  <= -1;
			num_outputs       <= 0;
			first_in_cycle    <= -1;
			first_out_cycle   <= -1;
			latency_cycles    <= -1;
			done_cycle        <= -1;
			measured_count    <= -1;
			start_cycle_tb    <= -1;
			elapsed_cycles_tb <= -1;
		end else begin
			cycle <= cycle + 1;
			// capture first input cycle when data_valid first asserts
			if (data_valid && first_in_cycle < 0)
				first_in_cycle <= cycle;
			// record TB-local start cycle at first data_valid
			if (data_valid && start_cycle_tb < 0) begin
				start_cycle_tb <= cycle;
			end

			if (softmax_valid) begin
				num_outputs      <= num_outputs + 1;
				last_valid_cycle <= cycle;
				if (first_valid_cycle < 0) first_valid_cycle <= cycle;
				// capture first output cycle and latency
				if (first_out_cycle < 0) begin
					first_out_cycle <= cycle;
					if (first_in_cycle >= 0) begin
						latency_cycles <= cycle - first_in_cycle;
						$display("[LAT] first_in=%0d -> first_out=%0d latency=%0d cycles", first_in_cycle, cycle, latency_cycles);
					end
				end
				$display("==== out_valid #%0d at cycle %0d (512b/32 lanes) ====", num_outputs+1, cycle);
				for (i_print = 0; i_print < CHUNK_LANES; i_print = i_print + 1) begin
					$display("out[%0d] = 0x%04h", i_print, softmax_out[i_print*16 +: 16]);
				end
				// 写入二进制输出（按 16bit 小端）
				for (i_print = 0; i_print < CHUNK_LANES; i_print = i_print + 1) begin
					word16 = softmax_out[i_print*16 +: 16];
					$fwrite(fd_out, "%c", word16[7:0]);
					$fwrite(fd_out, "%c", word16[15:8]);
				end
			end
			// 捕获 done 上升沿并保存 DUT 提供的 count（处理 done 可能只有一拍的情况）
			if (done && !prev_done && done_cycle < 0) begin
				done_cycle     <= cycle;
				measured_count <= count;
				// compute TB-local elapsed if we recorded a start
				if (start_cycle_tb >= 0) begin
					elapsed_cycles_tb = cycle - start_cycle_tb;
				end else begin
					elapsed_cycles_tb = -1;
				end
				$display("[DONE] done asserted at cycle %0d, DUT count=%0d, TB elapsed=%0d cycles", cycle, count, elapsed_cycles_tb);
			end
			// 更新 prev_done，用于上升沿检测
			prev_done <= done;
		end
	end

	// 主激励
	integer r;
	integer timeout;
	integer row_idx, chunk_idx, lane_idx;
	integer w_index, b_index;
	integer out_base;
	integer wait_timeout;
	// 文件 IO 句柄与缓冲
	integer fd_in;
	integer bytes_read;
	integer expect_words;
	integer expect_bytes;
	reg [7:0] in_bytes [0:2*N_ROWS*COLS_PER_ROW-1]; // 64*768 words * 2 bytes

	initial begin
		// 初始化
		data_in = {512{1'b0}};
		data_valid = 1'b0;
		rst = 1'b1;
		repeat (5) @(posedge clk);
		rst = 1'b0;

		// 读取 +args：+HOT=nnn +PATTERN=0/1 +READY_CONST=0/1 或 +READY_HIGH=H +READY_LOW=L
		begin : parse_plusargs
			integer tmp;
			if ($value$plusargs("HOT=%d", tmp)) begin
				g_hot_idx = tmp;
			end
			if ($value$plusargs("PATTERN=%d", tmp)) begin
				g_pattern = tmp;
			end
			if ($value$plusargs("READY_CONST=%d", tmp)) begin
				READY_CONST = tmp; // 0 或 1
			end
			if ($value$plusargs("READY_HIGH=%d", tmp)) begin
				READY_HIGH = tmp;  // >0 启用
			end
			if ($value$plusargs("READY_LOW=%d", tmp)) begin
				READY_LOW = tmp;   // >0 启用
			end
		end

		if (g_pattern == 1) begin
			compute_expect();
			$display("[TB] Config: PATTERN=%0d HOT_IDX=%0d -> round=%0d lane=%0d chunk=%0d",
					 g_pattern, g_hot_idx, hot_round, hot_lane32, hot_chunk);
		end else if (g_pattern == 2) begin
			$display("[TB] Config: PATTERN=%0d (per-round constants, 32 lanes x 24 rounds)", g_pattern);
		end else begin
			$display("[TB] Config: PATTERN=%0d (custom)", g_pattern);
		end

		// 读取输入 bin（小端）到缓存 a[row]
		expect_words = N_ROWS*COLS_PER_ROW; // 64*768
		expect_bytes = expect_words*2;
		fd_in = $fopen("C:/Users/ASUS/Desktop/acc/data/X_test_tensor_bf16.bin", "rb");
		if (fd_in == 0) begin
			$display("[ERROR] Cannot open input file: C:/Users/ASUS/Desktop/acc/data/X_test_tensor_bf16.bin");
			$finish;
		end
		bytes_read = $fread(in_bytes, fd_in);
		$fclose(fd_in);
		if (bytes_read != expect_bytes) begin
			$display("[ERROR] Read %0d bytes, expected %0d bytes", bytes_read, expect_bytes);
			$finish;
		end
		// 打包为 a[row][chunk][lane]
		for (row_idx = 0; row_idx < N_ROWS; row_idx = row_idx + 1) begin
			for (chunk_idx = 0; chunk_idx < ROUNDS_PER_ROW; chunk_idx = chunk_idx + 1) begin
				for (lane_idx = 0; lane_idx < 32; lane_idx = lane_idx + 1) begin
					w_index = row_idx*COLS_PER_ROW + chunk_idx*32 + lane_idx;
					b_index = w_index*2;
					word16[7:0]  = in_bytes[b_index + 0];
					word16[15:8] = in_bytes[b_index + 1];
					a[row_idx][chunk_idx*512 + lane_idx*16 +: 16] = word16;
				end
			end
		end
		$display("[TB] Loaded input X_test_tensor_bf16.bin into a[0..%0d]", N_ROWS-1);

		// 打开输出文件（小端 16bit 按序写入）
		fd_out = $fopen("C:/Users/ASUS/Desktop/acc/data/rmsnorm_out_tb.bin", "wb");
		if (fd_out == 0) begin
			$display("[ERROR] Cannot open output file: C:/Users/ASUS/Desktop/acc/data/rmsnorm_out_tb.bin");
			$finish;
		end

		// 连续流式发送：每周期输入 512bit，并同步产生 start 脉冲（start 在 always 块中跟随 data_valid）
		$display("[TB] Streaming all rows continuously with start pulses: %0d cycles", N_ROWS*ROUNDS_PER_ROW);
		for (row_idx = 0; row_idx < N_ROWS; row_idx = row_idx + 1) begin
			for (chunk_idx = 0; chunk_idx < ROUNDS_PER_ROW; chunk_idx = chunk_idx + 1) begin
				data_in    <= a[row_idx][chunk_idx*512 +: 512];
				data_valid <= 1'b1;
				@(posedge clk);
			end
		end
		data_valid <= 1'b0;

		// 等待所有输出轮（每拍 512b）或超时
		timeout = 0;
		while (num_outputs < EXPECT_CHUNKS && timeout < 2000000) begin
			@(posedge clk);
			timeout = timeout + 1;
		end

		$display("First output at cycle: %0d", first_valid_cycle);
		$display("Last  output at cycle: %0d", last_valid_cycle);
		$display("Total output rounds observed: %0d (expected %0d)", num_outputs, EXPECT_CHUNKS);
		if (first_in_cycle >= 0 && first_out_cycle >= 0) begin
			$display("[LAT][SUMMARY] first_in=%0d first_out=%0d latency=%0d cycles", first_in_cycle, first_out_cycle, latency_cycles);
		end else begin
			$display("[LAT][SUMMARY] latency not available (in=%0d, out=%0d)", first_in_cycle, first_out_cycle);
		end
		if (done_cycle >= 0) begin
			$display("[DONE][SUMMARY] done_cycle=%0d measured_count=%0d", done_cycle, measured_count);
		end else begin
			$display("[DONE][SUMMARY] done never asserted (done_cycle=%0d)", done_cycle);
		end
		if (num_outputs < EXPECT_CHUNKS) begin
			$display("TIMEOUT waiting for %0d rounds. Observed %0d.", EXPECT_CHUNKS, num_outputs);
		end

		$fclose(fd_out);
		repeat (10) @(posedge clk);
		$finish;
	end

endmodule
