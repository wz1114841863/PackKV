package briskkv

import chisel3._
import chisel3.util._

class VPacketReadRequest(countBits: Int) extends Bundle {
  val packIndex = UInt(countBits.W)
  val featureIndex = UInt(countBits.W)
}

class VPacketBufferStats extends Bundle {
  val activeCycles = UInt(64.W)
  val loadedPackets = UInt(64.W)
  val readRequests = UInt(64.W)
  val readResponses = UInt(64.W)
  val loadStallCycles = UInt(64.W)
  val responseStallCycles = UInt(64.W)
}

/** Values-only SRAM payload for one V descriptor.
  *
  * Packet metadata is validated before each write and reconstructed from the
  * read request plus transaction geometry. With the default parameters this
  * reduces an entry from 424 bits to 16 * 18 = 288 bits.
  */
class StoredVValues(valueBits: Int, packTokens: Int) extends Bundle {
  val values = Vec(packTokens, SInt(valueBits.W))
}

class VPacketBufferIO(
  valueBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val finish = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val tokenCount = Input(UInt(countBits.W))
  val loadIn = Flipped(
    Decoupled(new AttentionFeaturePacket(valueBits, packTokens, countBits))
  )
  val readRequest = Flipped(Decoupled(new VPacketReadRequest(countBits)))
  val readResponse = Decoupled(
    new AttentionFeaturePacket(valueBits, packTokens, countBits)
  )
  val loaded = Output(Bool())
  val loadDone = Output(Bool())
  val busy = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new VPacketBufferStats)
}

/** Stores the incoming pack-major/feature-major V stream for AV replay.
  *
  * The single synchronous read port is intentionally decoupled from loading.
  * Reads become legal only after the complete V tensor has been validated and
  * stored. One request may be outstanding and one response may be held under
  * backpressure.
  */
class VPacketBuffer(
  valueBits: Int = 18,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  require(valueBits >= 2)
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && blockTokens % packTokens == 0)
  require(maximumFeatureDim > 0 && maximumTokens >= packTokens)

  private val packShift = log2Ceil(packTokens)
  private val maximumPacks = (maximumTokens + packTokens - 1) / packTokens
  private val maximumDescriptors = maximumPacks * maximumFeatureDim
  private val addressBits = math.max(1, log2Ceil(maximumDescriptors))
  private val packsPerBlock = blockTokens / packTokens
  private val packsPerBlockShift = log2Ceil(packsPerBlock)
  private val packWithinBlockBits = math.max(1, packsPerBlockShift)
  require(isPow2(packsPerBlock))

  val io = IO(new VPacketBufferIO(valueBits, packTokens, countBits))

  val memory = SyncReadMem(
    maximumDescriptors,
    new StoredVValues(valueBits, packTokens)
  )

  val active = RegInit(false.B)
  val loadedReg = RegInit(false.B)
  val loadDoneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val featureDimReg = RegInit(0.U(countBits.W))
  val packCountReg = RegInit(0.U(countBits.W))
  val finalValidTokensReg = RegInit(0.U(log2Ceil(packTokens + 1).W))
  val expectedDescriptor = RegInit(0.U(countBits.W))
  val expectedPackIndex = RegInit(0.U(countBits.W))
  val expectedFeatureIndex = RegInit(0.U(countBits.W))

  val readOutstanding = RegInit(false.B)
  val responseValid = RegInit(false.B)
  val pendingPackIndex = RegInit(0.U(countBits.W))
  val pendingFeatureIndex = RegInit(0.U(countBits.W))
  val pendingDescriptorIndex = RegInit(0.U(countBits.W))
  val responsePacket = Reg(
    new AttentionFeaturePacket(valueBits, packTokens, countBits)
  )

  val activeCycles = RegInit(0.U(64.W))
  val loadedPackets = RegInit(0.U(64.W))
  val readRequests = RegInit(0.U(64.W))
  val readResponses = RegInit(0.U(64.W))
  val loadStallCycles = RegInit(0.U(64.W))
  val responseStallCycles = RegInit(0.U(64.W))

  loadDoneReg := false.B
  io.loaded := loadedReg
  io.loadDone := loadDoneReg
  io.busy := active || readOutstanding || responseValid
  io.error := errorReg
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.loadedPackets := Mux(enableStats.B, loadedPackets, 0.U)
  io.stats.readRequests := Mux(enableStats.B, readRequests, 0.U)
  io.stats.readResponses := Mux(enableStats.B, readResponses, 0.U)
  io.stats.loadStallCycles := Mux(enableStats.B, loadStallCycles, 0.U)
  io.stats.responseStallCycles := Mux(enableStats.B, responseStallCycles, 0.U)

  io.loadIn.ready := active && !loadedReg
  io.readRequest.ready := loadedReg && !readOutstanding && !responseValid
  io.readResponse.valid := responseValid
  io.readResponse.bits := responsePacket

  val loadFire = io.loadIn.valid && io.loadIn.ready
  val readRequestFire = io.readRequest.valid && io.readRequest.ready
  val readResponseFire = io.readResponse.valid && io.readResponse.ready

  val requestedPackValid = io.readRequest.bits.packIndex < packCountReg
  val requestedFeatureValid = io.readRequest.bits.featureIndex < featureDimReg
  val requestedAddress =
    io.readRequest.bits.packIndex * featureDimReg +
      io.readRequest.bits.featureIndex
  val safeReadAddress = Mux(
    requestedPackValid && requestedFeatureValid,
    requestedAddress,
    0.U
  )
  val readData = memory.read(
    safeReadAddress(addressBits - 1, 0),
    readRequestFire
  )
  val readDataPending = RegNext(readRequestFire, false.B)

  when(io.start && !active && !readOutstanding && !responseValid) {
    val parametersValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U &&
      io.tokenCount =/= 0.U && io.tokenCount <= maximumTokens.U
    active := parametersValid
    loadedReg := false.B
    errorReg := !parametersValid
    featureDimReg := io.featureDim
    packCountReg := (io.tokenCount + (packTokens - 1).U) >> packShift
    val remainder = io.tokenCount & (packTokens - 1).U
    finalValidTokensReg := Mux(remainder === 0.U, packTokens.U, remainder)
    expectedDescriptor := 0.U
    expectedPackIndex := 0.U
    expectedFeatureIndex := 0.U
    readOutstanding := false.B
    responseValid := false.B
    activeCycles := 0.U
    loadedPackets := 0.U
    readRequests := 0.U
    readResponses := 0.U
    loadStallCycles := 0.U
    responseStallCycles := 0.U
  }.elsewhen(io.start && (active || readOutstanding || responseValid)) {
    errorReg := true.B
  }.otherwise {
    when(active || readOutstanding || responseValid) {
      activeCycles := activeCycles + 1.U
    }
    when(active && !loadedReg && !io.loadIn.valid) {
      loadStallCycles := loadStallCycles + 1.U
    }
    when(io.readResponse.valid && !io.readResponse.ready) {
      responseStallCycles := responseStallCycles + 1.U
    }

    when(loadFire) {
      val expectedLast = expectedPackIndex === packCountReg - 1.U &&
        expectedFeatureIndex === featureDimReg - 1.U
      val expectedValidTokens = Mux(
        expectedPackIndex === packCountReg - 1.U,
        finalValidTokensReg,
        packTokens.U
      )
      val expectedBlockIndex = expectedPackIndex >> packsPerBlockShift
      val expectedPackWithinBlock = if (packsPerBlock == 1) {
        0.U(packWithinBlockBits.W)
      } else {
        expectedPackIndex(packWithinBlockBits - 1, 0)
      }
      val packetValid = io.loadIn.bits.descriptorIndex === expectedDescriptor &&
        io.loadIn.bits.packIndex === expectedPackIndex &&
        io.loadIn.bits.featureIndex === expectedFeatureIndex &&
        io.loadIn.bits.blockIndex === expectedBlockIndex &&
        io.loadIn.bits.packWithinBlock === expectedPackWithinBlock &&
        io.loadIn.bits.validTokens === expectedValidTokens &&
        io.loadIn.bits.last === expectedLast
      when(!packetValid) {
        errorReg := true.B
      }
      val storedValues = Wire(new StoredVValues(valueBits, packTokens))
      storedValues.values := io.loadIn.bits.values
      memory.write(expectedDescriptor(addressBits - 1, 0), storedValues)
      expectedDescriptor := expectedDescriptor + 1.U
      when(expectedFeatureIndex === featureDimReg - 1.U) {
        expectedFeatureIndex := 0.U
        expectedPackIndex := expectedPackIndex + 1.U
      }.otherwise {
        expectedFeatureIndex := expectedFeatureIndex + 1.U
      }
      loadedPackets := loadedPackets + 1.U
      when(expectedLast) {
        loadedReg := true.B
        loadDoneReg := true.B
      }
    }

    when(readRequestFire) {
      readOutstanding := true.B
      pendingPackIndex := io.readRequest.bits.packIndex
      pendingFeatureIndex := io.readRequest.bits.featureIndex
      pendingDescriptorIndex := requestedAddress
      readRequests := readRequests + 1.U
      when(!requestedPackValid || !requestedFeatureValid) {
        errorReg := true.B
      }
    }
    when(readDataPending) {
      val finalPack = pendingPackIndex === packCountReg - 1.U
      responsePacket.values := readData.values
      responsePacket.validTokens := Mux(
        finalPack,
        finalValidTokensReg,
        packTokens.U
      )
      responsePacket.descriptorIndex := pendingDescriptorIndex
      responsePacket.packIndex := pendingPackIndex
      responsePacket.featureIndex := pendingFeatureIndex
      responsePacket.blockIndex := pendingPackIndex >> packsPerBlockShift
      responsePacket.packWithinBlock := (if (packsPerBlock == 1) {
        0.U(packWithinBlockBits.W)
      } else {
        pendingPackIndex(packWithinBlockBits - 1, 0)
      })
      responsePacket.last := finalPack &&
        pendingFeatureIndex === featureDimReg - 1.U
      responseValid := true.B
      readOutstanding := false.B
    }
    when(readResponseFire) {
      responseValid := false.B
      readResponses := readResponses + 1.U
    }

    when(io.finish && active) {
      active := false.B
      loadedReg := false.B
      when(readOutstanding || responseValid) {
        errorReg := true.B
      }
    }.elsewhen(io.finish && !active) {
      errorReg := true.B
    }
  }
}
