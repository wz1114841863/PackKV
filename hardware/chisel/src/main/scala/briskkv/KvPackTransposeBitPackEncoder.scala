package briskkv

import chisel3._
import chisel3.util._

class KvPackTransposeBitPackEncoderIO(
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketIdBits: Int
) extends Bundle {
  private val format = BriskKvFormatV0.params
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val blockIndex = Input(UInt(countBits.W))
  val in = Flipped(
    Decoupled(
      new RoutedKvFeatureValue(
        format.kQuantBits,
        format.vQuantBits,
        countBits,
        tagBits,
        tokenIndexBits,
        bucketIdBits
      )
    )
  )
  val kMinimumOut = Decoupled(UInt(8.W))
  val kWidthOut = Decoupled(UInt(8.W))
  val kPayloadOut = Decoupled(UInt(8.W))
  val vMinimumOut = Decoupled(UInt(8.W))
  val vWidthOut = Decoupled(UInt(8.W))
  val vPayloadOut = Decoupled(UInt(8.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val transposeStats = Output(new KvPackTransposeStats)
  val kEncoderStats = Output(new DynamicBitPackEncoderStats)
  val vEncoderStats = Output(new DynamicBitPackEncoderStats)
}

/** Complete routed-q to K/V bit-packed component-stream pipeline. */
class KvPackTransposeBitPackEncoder(
  maximumFeatureDim: Int = 256,
  countBits: Int = 32,
  tagBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  private val format = BriskKvFormatV0.params
  private val tokenIndexBits = log2Ceil(format.blockTokens)
  private val descriptorCount = Wire(UInt(countBits.W))

  val io = IO(
    new KvPackTransposeBitPackEncoderIO(
      countBits,
      tagBits,
      tokenIndexBits,
      format.bucketIdBits
    )
  )

  val transpose = Module(
    new KvPackTransposeBuffer(
      maximumFeatureDim,
      countBits,
      tagBits,
      enableStats
    )
  )
  val kEncoder = Module(
    new DynamicBitPackEncoder(
      format.kQuantBits,
      format.packTokens,
      countBits,
      enableStats
    )
  )
  val vEncoder = Module(
    new DynamicBitPackEncoder(
      format.vQuantBits,
      format.packTokens,
      countBits,
      enableStats
    )
  )

  descriptorCount := io.featureDim << log2Ceil(
    format.blockTokens / format.packTokens
  )
  val commandValid = io.featureDim =/= 0.U &&
    io.featureDim <= maximumFeatureDim.U
  transpose.io.start := io.start
  transpose.io.featureDim := io.featureDim
  transpose.io.blockIndex := io.blockIndex
  transpose.io.in <> io.in
  kEncoder.io.start := io.start && commandValid
  kEncoder.io.descriptorCount := descriptorCount
  vEncoder.io.start := io.start && commandValid
  vEncoder.io.descriptorCount := descriptorCount

  val bothReady = kEncoder.io.in.ready && vEncoder.io.in.ready
  transpose.io.out.ready := bothReady
  kEncoder.io.in.valid := transpose.io.out.valid && vEncoder.io.in.ready
  vEncoder.io.in.valid := transpose.io.out.valid && kEncoder.io.in.ready
  kEncoder.io.in.bits.values := transpose.io.out.bits.kValues
  kEncoder.io.in.bits.descriptorIndex := transpose.io.out.bits.descriptorIndex
  kEncoder.io.in.bits.last := transpose.io.out.bits.last
  vEncoder.io.in.bits.values := transpose.io.out.bits.vValues
  vEncoder.io.in.bits.descriptorIndex := transpose.io.out.bits.descriptorIndex
  vEncoder.io.in.bits.last := transpose.io.out.bits.last

  io.kMinimumOut <> kEncoder.io.minimumOut
  io.kWidthOut <> kEncoder.io.widthOut
  io.kPayloadOut <> kEncoder.io.payloadOut
  io.vMinimumOut <> vEncoder.io.minimumOut
  io.vWidthOut <> vEncoder.io.widthOut
  io.vPayloadOut <> vEncoder.io.payloadOut

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val kDone = RegInit(false.B)
  val vDone = RegInit(false.B)
  doneReg := false.B
  when(io.start && !active) {
    active := commandValid
    doneReg := !commandValid
    kDone := false.B
    vDone := false.B
  }.elsewhen(active) {
    when(kEncoder.io.done) { kDone := true.B }
    when(vEncoder.io.done) { vDone := true.B }
    when((kDone || kEncoder.io.done) && (vDone || vEncoder.io.done)) {
      active := false.B
      doneReg := true.B
    }
  }

  io.busy := active || transpose.io.busy || kEncoder.io.busy || vEncoder.io.busy
  io.done := doneReg
  io.error := transpose.io.error || kEncoder.io.error || vEncoder.io.error
  io.transposeStats := transpose.io.stats
  io.kEncoderStats := kEncoder.io.stats
  io.vEncoderStats := vEncoder.io.stats
}
