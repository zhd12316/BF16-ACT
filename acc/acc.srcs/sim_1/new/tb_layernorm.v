`timescale 1ns/1ps 

// ==========================================
// LayerNorm 测试台模块
// ==========================================
// 功能：测试 final_top 模块的 LayerNorm 功能
// 输入：从文件读取 64行×768列 的 BF16 数据
// 输出：将归一化结果写入二进制文件
// ==========================================
module tb_layernorm;
    // ==========================================
    // 时钟和复位信号
    // ==========================================
    reg clk;  // 系统时钟，周期 10ns
    reg rst;  // 复位信号，高电平有效

    // ==========================================
    // DUT（待测设计）接口信号
    // ==========================================
    reg  [511:0]  data_in;      // 输入数据总线：每拍 512bit（32个 BF16 数）
    reg           data_valid;   // 输入有效信号：单周期脉冲，指示 data_in 有效
    wire [511:0]  softmax_out;  // 输出数据总线：每拍 512bit（32个 BF16 结果）
    wire          softmax_valid;// 输出有效信号：单周期脉冲，指示 softmax_out 有效
    reg           start;        // 启动信号：与 data_valid 同步拉高，指示新一轮计算开始
    wire          done;         // 完成信号：单周期脉冲，由 DUT 产生，指示当前计算完成
    wire [31:0]   count;        // 周期计数器：DUT 内部从 start 到 done 的时钟周期数

    // 测试控制信号
    reg           ready_in;     // 测试侧就绪信号：仅当 ready_in=1 时才发送下一轮输入
    
    // 操作模式参数：设置为 LayerNorm 模式
    parameter LAYERNORM_MODE = 3'b001;

    // ==========================================
    // 实例化待测模块（DUT）
    // ==========================================
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
        .mode(LAYERNORM_MODE)
    );

    // ==========================================
    // 时钟生成：10ns 周期（100MHz）
    // ==========================================
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ==========================================
    // 测试配置参数
    // ==========================================
    // 数据规模：
    // - 64 行数据
    // - 每行 768 个 BF16 数（32 lanes × 24 rounds）
    // - 每拍传输 512bit（32个 BF16）
    localparam integer N_ROWS            = 64;      // 总行数
    localparam integer COLS_PER_ROW      = 768;     // 每行列数（BF16 数量）
    localparam integer ROUNDS_PER_ROW    = 24;      // 每行需要的传输轮次（768/32=24）
    localparam integer OUTPUT_CHUNK_BITS = 512;     // 每拍输出位宽
    localparam integer CHUNK_LANES       = OUTPUT_CHUNK_BITS/16; // 每拍输出的 BF16 数量（32）
    localparam integer EXPECT_CHUNKS     = N_ROWS*ROUNDS_PER_ROW; // 期望的总输出轮次（64×24=1536）

    // ==========================================
    // 输入数据缓存
    // ==========================================
    // a[row][chunk_bits]: 存储每行的 24 轮输入数据
    // 每行占用 24×512 = 12,288 bits
    reg [512*ROUNDS_PER_ROW-1:0] a [0:N_ROWS-1];

    // ==========================================
    // 测试模式配置
    // ==========================================
    integer g_pattern = 1;      // 数据生成模式：
                                // 0 = 全 1.0
                                // 1 = one-hot（默认）
                                // 2 = 按轮次变化的常量
    integer g_hot_idx = 710;    // one-hot 模式下的热点索引（可通过 +HOT=nnn 覆盖）

    // ==========================================
    // BF16 常量定义
    // ==========================================
    localparam [15:0] BF16_ONE  = 16'h3F80; // BF16 格式的 1.0
    localparam [15:0] BF16_ZERO = 16'h0000; // BF16 格式的 0.0
    localparam [15:0] BF16_BIG  = 16'h4100; // BF16 格式的 8.0（用于 one-hot）

    // ==========================================
    // One-hot 模式辅助变量
    // ==========================================
    integer hot_round;   // 热点所在的轮次（0-23）
    integer hot_lane32;  // 热点所在的 lane（0-31）
    integer hot_chunk;   // 热点所在的输出块索引（与 hot_round 相同）

    // ==========================================
    // 任务：计算 one-hot 位置信息
    // ==========================================
    task compute_expect;
        begin
            hot_round   = g_hot_idx / 32;      // 热点在第几轮（索引/32）
            hot_lane32  = g_hot_idx % 32;      // 热点在该轮的第几个 lane
            hot_chunk   = hot_round;           // 输出块索引与轮次对齐
        end
    endtask

    // ==========================================
    // 函数：生成按轮次变化的 BF16 常量（用于模式 2）
    // ==========================================
    // 输入：轮次索引 r（0-23）
    // 输出：对应的 BF16 常量，范围 0.0 ~ 1.375
    // 策略：按 r%12 循环选择 12 个不同的值
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

    // ==========================================
    // 函数：生成一轮输入数据（512bit，32个 BF16）
    // ==========================================
    // 输入：round_idx - 轮次索引（0-23）
    // 输出：512bit 的输入数据
    // 根据 g_pattern 选择不同的生成策略
    function [511:0] gen_round_payload;
        input integer round_idx;
        integer i;
        integer hr;
        integer hl;
        reg [15:0] v_per_round;
        begin
            if (g_pattern == 0) begin
                // 模式 0：所有位置填充 1.0
                for (i = 0; i < 32; i = i + 1) begin
                    gen_round_payload[i*16 +: 16] = BF16_ONE;
                end
            end else if (g_pattern == 1) begin
                // 模式 1：one-hot - 仅 (hot_round, hot_lane) 位置为 BF16_BIG，其余为 0
                hr = g_hot_idx / 32;  // 热点所在轮次
                hl = g_hot_idx % 32;  // 热点所在 lane
                for (i = 0; i < 32; i = i + 1) begin
                    gen_round_payload[i*16 +: 16] = (round_idx==hr && i==hl) ? BF16_BIG : BF16_ZERO;
                end
            end else if (g_pattern == 2) begin
                // 模式 2：同一轮内 32 个 lane 相同，不同轮次取不同常量
                v_per_round = bf16_round_value(round_idx);
                for (i = 0; i < 32; i = i + 1) begin
                    gen_round_payload[i*16 +: 16] = v_per_round;
                end
            end else begin
                // 默认：全部初始化为 0
                for (i = 0; i < 32; i = i + 1) begin
                    gen_round_payload[i*16 +: 16] = BF16_ZERO;
                end
            end
        end
    endfunction

    // ==========================================
    // 任务：发送一轮数据
    // ==========================================
    // 功能：等待 ready_in 为高，然后在一个时钟周期内发送 512bit 数据
    // 输入：payload - 要发送的 512bit 数据
    task send_round;
        input [511:0] payload;
        begin
            // 等待 ready_in 变高
            while (ready_in == 1'b0) @(posedge clk);
            @(posedge clk);
            data_in <= payload;
            data_valid <= 1'b1;  // 拉高一个周期
            @(posedge clk);
            data_valid <= 1'b0;
        end
    endtask

    // ==========================================
    // ready_in 行为配置
    // ==========================================
    // 可通过 plusargs 配置三种模式：
    // 1. 常高/常低：READY_CONST=0/1
    // 2. 周期性抖动：READY_HIGH=H READY_LOW=L（高 H 周期，低 L 周期交替）
    // 3. 默认常高
    integer READY_CONST; // -1=未指定；0/1=常量值
    integer READY_HIGH;  // >0 时启用周期模式，高电平持续周期数
    integer READY_LOW;   // >0 时启用周期模式，低电平持续周期数
    integer ready_cnt;   // 周期计数器

    initial begin
        ready_in    = 1'b1;  // 默认常高
        READY_CONST = -1;    // -1 表示未指定常量
        READY_HIGH  = 0;     // 0 表示未启用周期模式
        READY_LOW   = 0;
        ready_cnt   = 0;
    end

    // ==========================================
    // ready_in 控制逻辑
    // ==========================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ready_cnt <= 0;
            if (READY_CONST != -1) 
                ready_in <= (READY_CONST != 0);  // 常量模式
            else 
                ready_in <= 1'b1;                 // 默认常高
        end else begin
            if (READY_CONST != -1) begin
                // 模式 1：常量值（0 或 1）
                ready_in <= (READY_CONST != 0);
            end else if (READY_HIGH > 0 && READY_LOW > 0) begin
                // 模式 2：周期性抖动
                // 保持 READY_HIGH 个周期为 1，然后 READY_LOW 个周期为 0，循环
                ready_cnt <= ready_cnt + 1;
                if (ready_in) begin
                    // 当前为高，检查是否达到高电平周期数
                    if (ready_cnt >= READY_HIGH) begin
                        ready_in  <= 1'b0;
                        ready_cnt <= 0;
                    end
                end else begin
                    // 当前为低，检查是否达到低电平周期数
                    if (ready_cnt >= READY_LOW) begin
                        ready_in  <= 1'b1;
                        ready_cnt <= 0;
                    end
                end
            end else begin
                // 模式 3：默认常高
                ready_in <= 1'b1;
            end
        end
    end

    // ==========================================
    // 输出监控和性能测量变量
    // ==========================================
    integer cycle;               // 当前周期数
    integer first_valid_cycle;   // 第一次输出有效的周期
    integer last_valid_cycle;    // 最后一次输出有效的周期
    integer num_outputs;         // 已观察到的输出轮次
    integer first_in_cycle;      // 第一次输入有效的周期
    integer first_out_cycle;     // 第一次输出有效的周期
    integer latency_cycles;      // 延迟：first_in -> first_out 的周期数
    integer i_print;             // 循环变量，用于打印

    // 输出文件和临时变量
    integer fd_out;              // 输出文件句柄
    reg [15:0] word16;           // 16bit 临时存储

    // done 信号相关
    integer done_cycle;          // done=1 的周期
    integer measured_count;      // DUT 提供的 count 值
    reg     prev_done;           // 上一拍的 done 值，用于检测上升沿
    integer start_cycle_tb;      // TB 记录的 start 周期（第一次 data_valid）
    integer elapsed_cycles_tb;   // TB 计算的 start->done 周期数

    // 延迟副本，用于边沿检测
    reg data_valid_d;

    initial cycle = 0;

    // ==========================================
    // start 信号生成逻辑
    // ==========================================
    // 当 data_valid 从 0->1 跳变时，生成单周期的 start 脉冲
    always @(posedge clk) begin
        if (rst) begin
            start <= 1'b0;
            data_valid_d <= 1'b0;
        end else begin
            start <= data_valid && ~data_valid_d;  // 上升沿检测
            data_valid_d <= data_valid;
        end
    end

    // ==========================================
    // 输出监控和性能统计逻辑
    // ==========================================
    always @(posedge clk) begin
        if (rst) begin
            // 复位时初始化所有统计变量
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

            // 捕获第一次输入有效的周期
            if (data_valid && first_in_cycle < 0)
                first_in_cycle <= cycle;

            // 记录 TB 本地的 start 周期（第一次 data_valid）
            if (data_valid && start_cycle_tb < 0) begin
                start_cycle_tb <= cycle;
            end

            // 输出有效时的处理
            if (softmax_valid) begin
                num_outputs      <= num_outputs + 1;
                last_valid_cycle <= cycle;
                if (first_valid_cycle < 0) first_valid_cycle <= cycle;

                // 捕获第一次输出周期并计算延迟
                if (first_out_cycle < 0) begin
                    first_out_cycle <= cycle;
                    if (first_in_cycle >= 0) begin
                        latency_cycles <= cycle - first_in_cycle;
                        $display("[LAT] first_in=%0d -> first_out=%0d latency=%0d cycles", 
                                first_in_cycle, cycle, latency_cycles);
                    end
                end

                // 打印当前输出的 32 个 BF16 值
                $display("==== out_valid #%0d at cycle %0d (512b/32 lanes) ====", 
                        num_outputs+1, cycle);
                for (i_print = 0; i_print < CHUNK_LANES; i_print = i_print + 1) begin
                    $display("out[%0d] = 0x%04h", i_print, softmax_out[i_print*16 +: 16]);
                end

                // 写入二进制输出文件（小端序，每个 BF16 占 2 字节）
                for (i_print = 0; i_print < CHUNK_LANES; i_print = i_print + 1) begin
                    word16 = softmax_out[i_print*16 +: 16];
                    $fwrite(fd_out, "%c", word16[7:0]);   // 低字节
                    $fwrite(fd_out, "%c", word16[15:8]);  // 高字节
                end
            end

            // 捕获 done 上升沿（处理 done 可能只持续 1 拍的情况）
            if (done && !prev_done && done_cycle < 0) begin
                done_cycle     <= cycle;
                measured_count <= count;
                // 计算 TB 本地的 elapsed 时间
                if (start_cycle_tb >= 0) begin
                    elapsed_cycles_tb = cycle - start_cycle_tb;
                end else begin
                    elapsed_cycles_tb = -1;
                end
                $display("[DONE] done asserted at cycle %0d, DUT count=%0d, TB elapsed=%0d cycles", 
                        cycle, count, elapsed_cycles_tb);
            end

            // 更新 prev_done 以便下次检测上升沿
            prev_done <= done;
        end
    end

    // ==========================================
    // 主激励流程
    // ==========================================
    integer r;
    integer timeout;
    integer row_idx, chunk_idx, lane_idx;
    integer w_index, b_index;
    integer out_base;
    integer wait_timeout;

    // 文件 IO 相关
    integer fd_in;               // 输入文件句柄
    integer bytes_read;          // 实际读取的字节数
    integer expect_words;        // 期望的 BF16 数量
    integer expect_bytes;        // 期望的字节数
    reg [7:0] in_bytes [0:2*N_ROWS*COLS_PER_ROW-1]; // 输入缓存：64×768 words × 2 bytes

    initial begin
        // ==========================================
        // 初始化
        // ==========================================
        data_in = {512{1'b0}};
        data_valid = 1'b0;
        rst = 1'b1;
        repeat (5) @(posedge clk);
        rst = 1'b0;

        // ==========================================
        // 解析命令行参数
        // ==========================================
        // +HOT=nnn        : 设置 one-hot 索引
        // +PATTERN=0/1/2  : 设置数据生成模式
        // +READY_CONST=0/1: 设置 ready_in 为常量
        // +READY_HIGH=H   : 周期模式高电平周期数
        // +READY_LOW=L    : 周期模式低电平周期数
        begin : parse_plusargs
            integer tmp;
            if ($value$plusargs("HOT=%d", tmp)) begin
                g_hot_idx = tmp;
            end
            if ($value$plusargs("PATTERN=%d", tmp)) begin
                g_pattern = tmp;
            end
            if ($value$plusargs("READY_CONST=%d", tmp)) begin
                READY_CONST = tmp;
            end
            if ($value$plusargs("READY_HIGH=%d", tmp)) begin
                READY_HIGH = tmp;
            end
            if ($value$plusargs("READY_LOW=%d", tmp)) begin
                READY_LOW = tmp;
            end
        end

        // ==========================================
        // 打印测试配置
        // ==========================================
        if (g_pattern == 1) begin
            compute_expect();
            $display("[TB] Config: PATTERN=%0d HOT_IDX=%0d -> round=%0d lane=%0d chunk=%0d",
                     g_pattern, g_hot_idx, hot_round, hot_lane32, hot_chunk);
        end else if (g_pattern == 2) begin
            $display("[TB] Config: PATTERN=%0d (per-round constants, 32 lanes x 24 rounds)", g_pattern);
        end else begin
            $display("[TB] Config: PATTERN=%0d (custom)", g_pattern);
        end

        // ==========================================
        // 读取输入数据文件
        // ==========================================
        expect_words = N_ROWS*COLS_PER_ROW;  // 64×768 = 49,152 个 BF16
        expect_bytes = expect_words*2;        // 98,304 字节
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

        // ==========================================
        // 将读取的字节数据打包到 a[row][chunk][lane]
        // ==========================================
        // 数据组织：a[row_idx][chunk_idx*512 + lane_idx*16 +: 16]
        // - row_idx: 0-63（行索引）
        // - chunk_idx: 0-23（每行的轮次索引）
        // - lane_idx: 0-31（每轮的 lane 索引）
        for (row_idx = 0; row_idx < N_ROWS; row_idx = row_idx + 1) begin
            for (chunk_idx = 0; chunk_idx < ROUNDS_PER_ROW; chunk_idx = chunk_idx + 1) begin
                for (lane_idx = 0; lane_idx < 32; lane_idx = lane_idx + 1) begin
                    w_index = row_idx*COLS_PER_ROW + chunk_idx*32 + lane_idx;
                    b_index = w_index*2;
                    word16[7:0]  = in_bytes[b_index + 0];  // 低字节
                    word16[15:8] = in_bytes[b_index + 1];  // 高字节
                    a[row_idx][chunk_idx*512 + lane_idx*16 +: 16] = word16;
                end
            end
        end
        $display("[TB] Loaded input X_test_tensor_bf16.bin into a[0..%0d]", N_ROWS-1);

        // ==========================================
        // 打开输出文件
        // ==========================================
        fd_out = $fopen("C:/Users/ASUS/Desktop/acc/data/layernorm_out_tb.bin", "wb");
        if (fd_out == 0) begin
            $display("[ERROR] Cannot open output file: C:/Users/ASUS/Desktop/acc/data/layernorm_out_tb.bin");
            $finish;
        end

        // ==========================================
        // 连续流式发送所有数据
        // ==========================================
        // 总共发送 64行×24轮 = 1536 轮
        // 每轮发送 512bit（32个 BF16）
        // start 脉冲由 always 块自动生成（跟随 data_valid 上升沿）
        $display("[TB] Streaming all rows continuously with start pulses: %0d cycles", N_ROWS*ROUNDS_PER_ROW);
        for (row_idx = 0; row_idx < N_ROWS; row_idx = row_idx + 1) begin
            for (chunk_idx = 0; chunk_idx < ROUNDS_PER_ROW; chunk_idx = chunk_idx + 1) begin
                data_in    <= a[row_idx][chunk_idx*512 +: 512];
                data_valid <= 1'b1;
                @(posedge clk);
            end
        end
        data_valid <= 1'b0;

        // ==========================================
        // 等待所有输出或超时
        // ==========================================
        timeout = 0;
        while (num_outputs < EXPECT_CHUNKS && timeout < 2000000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        // ==========================================
        // 打印测试总结
        // ==========================================
        $display("First output at cycle: %0d", first_valid_cycle);
        $display("Last  output at cycle: %0d", last_valid_cycle);
        $display("Total output rounds observed: %0d (expected %0d)", num_outputs, EXPECT_CHUNKS);
        
        if (first_in_cycle >= 0 && first_out_cycle >= 0) begin
            $display("[LAT][SUMMARY] first_in=%0d first_out=%0d latency=%0d cycles", 
                    first_in_cycle, first_out_cycle, latency_cycles);
        end else begin
            $display("[LAT][SUMMARY] latency not available (in=%0d, out=%0d)", 
                    first_in_cycle, first_out_cycle);
        end
        
        if (done_cycle >= 0) begin
            $display("[DONE][SUMMARY] done_cycle=%0d measured_count=%0d", 
                    done_cycle, measured_count);
        end else begin
            $display("[DONE][SUMMARY] done never asserted (done_cycle=%0d)", done_cycle);
        end
        
        if (num_outputs < EXPECT_CHUNKS) begin
            $display("TIMEOUT waiting for %0d rounds. Observed %0d.", EXPECT_CHUNKS, num_outputs);
        end

        // ==========================================
        // 清理和结束
        // ==========================================
        $fclose(fd_out);
        repeat (10) @(posedge clk);
        $finish;
    end

endmodule
