`timescale 1ns/1ps

/**
 * 64-token end-to-end smoke/overlap test for the generated full JIT-V tile.
 *
 * This is the SystemVerilog counterpart of
 * BriskKvJitVSingleHeadTileTopSpec. It exercises raw K/V write, quantization,
 * bucket routing, bit packing, resident stream memories, decompression,
 * fixed-point attention, output backpressure, and JIT-V overlap.
 */
module tb_briskkv_jit_v;
  localparam int FEATURE_DIM = 4;
  localparam int TOKEN_COUNT = 64;
  localparam int PACK_COUNT = TOKEN_COUNT / 16;
  localparam int PAIR_COUNT = TOKEN_COUNT * FEATURE_DIM;
  localparam int MAX_WRITE_CYCLES = 100000;
  localparam int MAX_ATTENTION_CYCLES = 200000;

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
  wire  io_writeDone;
  wire  io_encodedReady;
  wire  io_busy;
  wire  io_error;
  wire  io_attentionProgress_vLaunched;

  integer expected_output [0:FEATURE_DIM-1];
  integer query [0:FEATURE_DIM-1];
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
  bit overlap_seen;
  bit result_seen;
  string wave_file;

  BriskKvJitVSingleHeadTileTop dut (
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
    .io_writeDone(io_writeDone),
    .io_encodedReady(io_encodedReady),
    .io_busy(io_busy),
    .io_error(io_error),
    .io_attentionProgress_vLaunched(io_attentionProgress_vLaunched)
  );

  initial clock = 1'b0;
  always #1.0 clock = ~clock; // 2.0 ns / 500 MHz reference clock

  initial begin
    wave_file = "outputs/jit_v_overlap_64t";
    void'($value$plusargs("WAVE_FILE=%s", wave_file));
    if ($test$plusargs("VCD")) begin
      $dumpfile({wave_file, ".vcd"});
      $dumpvars(0, tb_briskkv_jit_v.dut);
    end
`ifdef BRISKKV_VCS
    if ($test$plusargs("VPD")) begin
      $vcdplusfile({wave_file, ".vpd"});
      $vcdpluson(0, tb_briskkv_jit_v.dut);
    end
`endif
  end

  task automatic drive_write_pair(input integer pair);
    begin
      token_index = pair / FEATURE_DIM;
      feature_index = pair % FEATURE_DIM;
      case (feature_index)
        0: begin
          k_quarter_units = 0;
          v_quarter_units = 0;
        end
        1: begin
          k_quarter_units = token_index % 4;
          v_quarter_units = (token_index + 1) % 4;
        end
        2: begin
          k_quarter_units = (token_index / 4) % 4;
          v_quarter_units = ((token_index / 4) + 2) % 4;
        end
        default: begin
          k_quarter_units = 4;
          v_quarter_units = 4;
        end
      endcase
      // Q12 raw input: one quarter unit is 4096 / 4 = 1024.
      io_writeIn_bits_kFixedRaw = k_quarter_units * 1024;
      io_writeIn_bits_vFixedRaw = v_quarter_units * 1024;
      io_writeIn_bits_tokenTag = token_index;
      io_writeIn_bits_blockIndex = 0;
      io_writeIn_bits_tokenIndex = token_index;
      io_writeIn_bits_featureIndex = feature_index;
      io_writeIn_bits_lastFeature = feature_index == FEATURE_DIM - 1;
      io_writeIn_bits_last = pair == PAIR_COUNT - 1;
      io_writeIn_valid = 1'b1;
    end
  endtask

  initial begin
    expected_output[0] = 0;
    expected_output[1] = 24;
    expected_output[2] = 24;
    expected_output[3] = 64;
    query[0] = 8;
    query[1] = -5;
    query[2] = 3;
    query[3] = 6;

    reset = 1'b1;
    io_writeStart = 1'b0;
    io_featureDim = FEATURE_DIM;
    io_blockCount = 1;
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
    io_attentionTag = 16'd121;
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

    // Start and feed one 64-token block in token-major/feature-minor order.
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
    if (write_cycles >= MAX_WRITE_CYCLES)
      $fatal(1, "write timeout after %0d cycles", write_cycles);
    if (pair_index != PAIR_COUNT)
      $fatal(1, "accepted K/V pairs mismatch: got=%0d expected=%0d",
             pair_index, PAIR_COUNT);
    if (io_error || !io_encodedReady)
      $fatal(1, "write failed: io_error=%0b encodedReady=%0b",
             io_error, io_encodedReady);

    // Launch resident-cache attention.
    while (!io_attentionReady) @(posedge clock);
    @(negedge clock);
    io_attentionStart = 1'b1;
    @(posedge clock);
    @(negedge clock);
    io_attentionStart = 1'b0;
    io_result_ready = 1'b1;

    query_index = 0;
    output_index = 0;
    attention_cycles = 0;
    overlap_seen = 1'b0;
    result_seen = 1'b0;
    while (!result_seen && attention_cycles < MAX_ATTENTION_CYCLES) begin
      if (query_index < FEATURE_DIM) begin
        io_queryIn_valid = 1'b1;
        io_queryIn_bits = query[query_index];
      end else begin
        io_queryIn_valid = 1'b0;
      end
      // Periodic downstream backpressure matches the Chisel test.
      io_attentionOut_ready = (attention_cycles % 5) != 2;

      @(posedge clock);
      if (io_queryIn_valid && io_queryIn_ready)
        query_index = query_index + 1;

      if (io_attentionProgress_vLaunched &&
          dut.attention.jitV.weightLoadIndex < PACK_COUNT)
        overlap_seen = 1'b1;

      if (io_attentionOut_valid && io_attentionOut_ready) begin
        if (output_index >= FEATURE_DIM)
          $fatal(1, "received too many attention outputs");
        if (io_attentionOut_bits_featureIndex !== output_index)
          $fatal(1, "feature index mismatch: got=%0d expected=%0d",
                 io_attentionOut_bits_featureIndex, output_index);
        observed_value = $signed(io_attentionOut_bits_value);
        if (observed_value != expected_output[output_index])
          $fatal(1, "attention output[%0d] mismatch: got=%0d expected=%0d",
                 output_index, observed_value, expected_output[output_index]);
        if (io_attentionOut_bits_last !== (output_index == FEATURE_DIM - 1))
          $fatal(1, "attention last flag mismatch at output %0d", output_index);
        output_index = output_index + 1;
      end

      if (io_result_valid && io_result_ready) begin
        if (io_result_bits_tag !== 16'd121)
          $fatal(1, "result tag mismatch: got=%0d", io_result_bits_tag);
        if (io_result_bits_error)
          $fatal(1, "result.error asserted");
        result_seen = 1'b1;
      end
      @(negedge clock);
      attention_cycles = attention_cycles + 1;
    end

    if (!result_seen)
      $fatal(1, "attention timeout after %0d cycles", attention_cycles);
    if (query_index != FEATURE_DIM)
      $fatal(1, "query count mismatch: got=%0d expected=%0d",
             query_index, FEATURE_DIM);
    if (output_index != FEATURE_DIM)
      $fatal(1, "output count mismatch: got=%0d expected=%0d",
             output_index, FEATURE_DIM);
    if (!overlap_seen)
      $fatal(1, "JIT-V overlap was not observed");
    if (io_error || !io_encodedReady)
      $fatal(1, "tile ended in an invalid state: error=%0b encodedReady=%0b",
             io_error, io_encodedReady);

    $display("BRISK-KV VCS PASS: write_cycles=%0d attention_cycles=%0d overlap=%0b",
             write_cycles, attention_cycles, overlap_seen);
`ifdef BRISKKV_VCS
    if ($test$plusargs("VPD"))
      $vcdplusoff;
`endif
    repeat (5) @(posedge clock);
    $finish;
  end
endmodule
