`timescale 1ns/1ps

`ifndef BRISKKV_DUT_TOP
  `error "BRISKKV_DUT_TOP must name the generated unified-tile top"
`endif

/** Matched Full-V/shared 1024-token x 128-feature activity workload. */
module tb_briskkv_tile_power_1024;
  localparam int FEATURE_DIM = 128;
  localparam int TOKEN_COUNT = 1024;
  localparam int BLOCK_TOKENS = 64;
  localparam int BLOCK_COUNT = TOKEN_COUNT / BLOCK_TOKENS;
  localparam int PACK_COUNT = TOKEN_COUNT / 16;
  localparam int PAIR_COUNT = TOKEN_COUNT * FEATURE_DIM;
  localparam int MAX_WRITE_CYCLES = 4000000;
  localparam int MAX_ATTENTION_CYCLES = 4000000;
  localparam longint signed EXPECTED_OUTPUT_CHECKSUM =
    64'sd6216823619359318016;

  logic clock;
  logic reset;
  logic io_writeStart;
  wire  io_writeReady;
  logic [31:0] io_featureDim;
  logic [31:0] io_blockCount;
  logic [31:0] io_firstBlockIndex;
  wire  io_writeIn_ready;
  logic io_writeIn_valid;
  logic [23:0] io_writeIn_bits_kFixedRaw;
  logic [23:0] io_writeIn_bits_vFixedRaw;
  logic [15:0] io_writeIn_bits_tokenTag;
  logic [31:0] io_writeIn_bits_blockIndex;
  logic [5:0] io_writeIn_bits_tokenIndex;
  logic [31:0] io_writeIn_bits_featureIndex;
  logic io_writeIn_bits_lastFeature;
  logic io_writeIn_bits_last;
  logic io_attentionStart;
  wire  io_attentionReady;
  logic [15:0] io_attentionTag;
  wire  io_queryIn_ready;
  logic io_queryIn_valid;
  logic signed [17:0] io_queryIn_bits;
  logic io_bucketOut_ready;
  logic io_attentionOut_ready;
  wire  io_attentionOut_valid;
  wire signed [17:0] io_attentionOut_bits_value;
  wire [31:0] io_attentionOut_bits_featureIndex;
  wire  io_attentionOut_bits_last;
  logic io_result_ready;
  wire  io_result_valid;
  wire [15:0] io_result_bits_tag;
  wire  io_result_bits_error;
  wire [31:0] io_result_bits_tokenCount;
  wire [31:0] io_result_bits_packCount;
  wire [31:0] io_result_bits_blockCount;
  wire  io_writeDone;
  wire  io_encodedReady;
  wire  io_busy;
  wire  io_error;

  integer pair_index;
  integer query_index;
  integer output_index;
  integer write_cycles;
  integer attention_cycles;
  integer token_index;
  integer feature_index;
  integer k_quarter_units;
  integer v_quarter_units;
  integer observed_value;
  longint signed output_checksum;
  bit result_seen;
  bit activity_running;
  time activity_start_time;
  time activity_stop_time;
  string wave_file;
  string activity_phase;

  `BRISKKV_DUT_TOP dut (
    .clock(clock),
    .reset(reset),
    .io_writeStart(io_writeStart),
    .io_writeReady(io_writeReady),
    .io_featureDim(io_featureDim),
    .io_blockCount(io_blockCount),
    .io_firstBlockIndex(io_firstBlockIndex),
    .io_writeIn_ready(io_writeIn_ready),
    .io_writeIn_valid(io_writeIn_valid),
    .io_writeIn_bits_kFixedRaw(io_writeIn_bits_kFixedRaw),
    .io_writeIn_bits_vFixedRaw(io_writeIn_bits_vFixedRaw),
    .io_writeIn_bits_tokenTag(io_writeIn_bits_tokenTag),
    .io_writeIn_bits_blockIndex(io_writeIn_bits_blockIndex),
    .io_writeIn_bits_tokenIndex(io_writeIn_bits_tokenIndex),
    .io_writeIn_bits_featureIndex(io_writeIn_bits_featureIndex),
    .io_writeIn_bits_lastFeature(io_writeIn_bits_lastFeature),
    .io_writeIn_bits_last(io_writeIn_bits_last),
    .io_attentionStart(io_attentionStart),
    .io_attentionReady(io_attentionReady),
    .io_attentionTag(io_attentionTag),
    .io_queryIn_ready(io_queryIn_ready),
    .io_queryIn_valid(io_queryIn_valid),
    .io_queryIn_bits(io_queryIn_bits),
    .io_bucketOut_ready(io_bucketOut_ready),
    .io_attentionOut_ready(io_attentionOut_ready),
    .io_attentionOut_valid(io_attentionOut_valid),
    .io_attentionOut_bits_value(io_attentionOut_bits_value),
    .io_attentionOut_bits_featureIndex(io_attentionOut_bits_featureIndex),
    .io_attentionOut_bits_last(io_attentionOut_bits_last),
    .io_result_ready(io_result_ready),
    .io_result_valid(io_result_valid),
    .io_result_bits_tag(io_result_bits_tag),
    .io_result_bits_error(io_result_bits_error),
    .io_result_bits_tokenCount(io_result_bits_tokenCount),
    .io_result_bits_packCount(io_result_bits_packCount),
    .io_result_bits_blockCount(io_result_bits_blockCount),
    .io_writeDone(io_writeDone),
    .io_encodedReady(io_encodedReady),
    .io_busy(io_busy),
    .io_error(io_error)
  );

  initial clock = 1'b0;
  always #1.0 clock = ~clock;

  initial begin
    wave_file = "outputs/briskkv_tile_1024t_128f_attention";
    activity_phase = "attention";
    void'($value$plusargs("WAVE_FILE=%s", wave_file));
    void'($value$plusargs("ACTIVITY_PHASE=%s", activity_phase));
    if (activity_phase != "attention" && activity_phase != "write" &&
        activity_phase != "combined")
      $fatal(1, "unsupported ACTIVITY_PHASE=%s", activity_phase);
    if ($test$plusargs("VCD"))
      $dumpfile({wave_file, ".vcd"});
`ifdef BRISKKV_VCS
    if ($test$plusargs("VPD"))
      $vcdplusfile({wave_file, ".vpd"});
`endif
  end

  task automatic activity_start(input string phase);
    begin
      if ((activity_phase == phase || activity_phase == "combined") &&
          !activity_running) begin
        if ($test$plusargs("VCD"))
          $dumpvars(0, tb_briskkv_tile_power_1024.dut);
`ifdef BRISKKV_VCS
        if ($test$plusargs("VPD"))
          $vcdpluson(0, tb_briskkv_tile_power_1024.dut);
`endif
        activity_running = 1'b1;
        activity_start_time = $time;
      end
    end
  endtask

  task automatic activity_stop(input string phase);
    begin
      if (activity_phase == phase) begin
        if ($test$plusargs("VCD"))
          $dumpoff;
`ifdef BRISKKV_VCS
        if ($test$plusargs("VPD"))
          $vcdplusoff;
`endif
        activity_running = 1'b0;
        activity_stop_time = $time;
      end
    end
  endtask

  task automatic drive_write_pair(input integer pair);
    begin
      token_index = pair / FEATURE_DIM;
      feature_index = pair % FEATURE_DIM;
      k_quarter_units = (token_index + feature_index * 3) % 5;
      v_quarter_units = (token_index * 3 + feature_index * 2 + 1) % 5;
      io_writeIn_bits_kFixedRaw = k_quarter_units * 1024;
      io_writeIn_bits_vFixedRaw = v_quarter_units * 1024;
      io_writeIn_bits_tokenTag = token_index;
      io_writeIn_bits_blockIndex = token_index / BLOCK_TOKENS;
      io_writeIn_bits_tokenIndex = token_index % BLOCK_TOKENS;
      io_writeIn_bits_featureIndex = feature_index;
      io_writeIn_bits_lastFeature = feature_index == FEATURE_DIM - 1;
      io_writeIn_bits_last = pair == PAIR_COUNT - 1;
      io_writeIn_valid = 1'b1;
    end
  endtask

  initial begin
    reset = 1'b1;
    activity_running = 1'b0;
    activity_start_time = 0;
    activity_stop_time = 0;
    io_writeStart = 1'b0;
    io_featureDim = FEATURE_DIM;
    io_blockCount = BLOCK_COUNT;
    io_firstBlockIndex = 0;
    io_writeIn_valid = 1'b0;
    io_writeIn_bits_kFixedRaw = '0;
    io_writeIn_bits_vFixedRaw = '0;
    io_writeIn_bits_tokenTag = '0;
    io_writeIn_bits_blockIndex = '0;
    io_writeIn_bits_tokenIndex = '0;
    io_writeIn_bits_featureIndex = '0;
    io_writeIn_bits_lastFeature = 1'b0;
    io_writeIn_bits_last = 1'b0;
    io_attentionStart = 1'b0;
    io_attentionTag = 16'd1024;
    io_queryIn_valid = 1'b0;
    io_queryIn_bits = '0;
    io_bucketOut_ready = 1'b1;
    io_attentionOut_ready = 1'b0;
    io_result_ready = 1'b0;

    repeat (5) @(posedge clock);
    @(negedge clock);
    reset = 1'b0;
    if (!io_writeReady)
      $fatal(1, "write interface was not ready after reset");

    activity_start("write");
    io_writeStart = 1'b1;
    @(posedge clock);
    @(negedge clock);
    io_writeStart = 1'b0;
    pair_index = 0;
    write_cycles = 0;
    while (!io_writeDone && write_cycles < MAX_WRITE_CYCLES) begin
      if (pair_index < PAIR_COUNT)
        drive_write_pair(pair_index);
      else
        io_writeIn_valid = 1'b0;
      @(posedge clock);
      if (io_writeIn_valid && io_writeIn_ready)
        pair_index = pair_index + 1;
      @(negedge clock);
      write_cycles = write_cycles + 1;
    end
    io_writeIn_valid = 1'b0;
    activity_stop("write");
    if (write_cycles >= MAX_WRITE_CYCLES)
      $fatal(1, "write timeout after %0d cycles", write_cycles);
    if (pair_index != PAIR_COUNT)
      $fatal(1, "accepted K/V pairs mismatch: got=%0d expected=%0d",
             pair_index, PAIR_COUNT);
    if (io_error || !io_encodedReady)
      $fatal(1, "write failed: io_error=%0b encodedReady=%0b",
             io_error, io_encodedReady);

    while (!io_attentionReady) @(posedge clock);
    @(negedge clock);
    activity_start("attention");
    io_attentionStart = 1'b1;
    @(posedge clock);
    @(negedge clock);
    io_attentionStart = 1'b0;
    io_result_ready = 1'b1;

    query_index = 0;
    output_index = 0;
    output_checksum = 0;
    attention_cycles = 0;
    result_seen = 1'b0;
    while (!result_seen && attention_cycles < MAX_ATTENTION_CYCLES) begin
      if (query_index < FEATURE_DIM) begin
        io_queryIn_valid = 1'b1;
        io_queryIn_bits = ((query_index * 5 + 3) % 11) - 5;
      end else begin
        io_queryIn_valid = 1'b0;
      end
      io_attentionOut_ready = (attention_cycles % 4) == 0;

      @(posedge clock);
      if (io_queryIn_valid && io_queryIn_ready)
        query_index = query_index + 1;
      if (io_attentionOut_valid && io_attentionOut_ready) begin
        if (output_index >= FEATURE_DIM)
          $fatal(1, "received too many attention outputs");
        if (io_attentionOut_bits_featureIndex !== output_index)
          $fatal(1, "feature index mismatch: got=%0d expected=%0d",
                 io_attentionOut_bits_featureIndex, output_index);
        observed_value = $signed(io_attentionOut_bits_value);
        output_checksum = output_checksum * 131 + observed_value;
        if (io_attentionOut_bits_last !== (output_index == FEATURE_DIM - 1))
          $fatal(1, "attention last flag mismatch at output %0d", output_index);
        output_index = output_index + 1;
      end

      if (io_result_valid && io_result_ready) begin
        if (io_result_bits_tag !== 16'd1024)
          $fatal(1, "result tag mismatch: got=%0d", io_result_bits_tag);
        if (io_result_bits_error)
          $fatal(1, "result.error asserted");
        if (io_result_bits_tokenCount !== TOKEN_COUNT ||
            io_result_bits_packCount !== PACK_COUNT ||
            io_result_bits_blockCount !== BLOCK_COUNT)
          $fatal(1, "result geometry mismatch: tokens=%0d packs=%0d blocks=%0d",
                 io_result_bits_tokenCount, io_result_bits_packCount,
                 io_result_bits_blockCount);
        result_seen = 1'b1;
      end
      @(negedge clock);
      attention_cycles = attention_cycles + 1;
    end
    activity_stop("attention");

    if (!result_seen)
      $fatal(1, "attention timeout after %0d cycles", attention_cycles);
    if (query_index != FEATURE_DIM)
      $fatal(1, "query count mismatch: got=%0d expected=%0d",
             query_index, FEATURE_DIM);
    if (output_index != FEATURE_DIM)
      $fatal(1, "output count mismatch: got=%0d expected=%0d",
             output_index, FEATURE_DIM);
    if (output_checksum != EXPECTED_OUTPUT_CHECKSUM)
      $fatal(1, "output checksum mismatch: got=%0d expected=%0d",
             output_checksum, EXPECTED_OUTPUT_CHECKSUM);
    if (io_error || !io_encodedReady)
      $fatal(1, "tile ended in an invalid state: error=%0b encodedReady=%0b",
             io_error, io_encodedReady);

    if (activity_running) begin
      if ($test$plusargs("VCD"))
        $dumpoff;
`ifdef BRISKKV_VCS
      if ($test$plusargs("VPD"))
        $vcdplusoff;
`endif
      activity_running = 1'b0;
      activity_stop_time = $time;
    end
    $display("BRISK-KV TILE 1024x128 POWER PASS: phase=%s write_cycles=%0d attention_cycles=%0d checksum=%0d activity_start_ns=%0d activity_stop_ns=%0d activity_duration_ns=%0d",
             activity_phase, write_cycles, attention_cycles, output_checksum,
             activity_start_time, activity_stop_time,
             activity_stop_time - activity_start_time);
    repeat (5) @(posedge clock);
    $finish;
  end
endmodule
