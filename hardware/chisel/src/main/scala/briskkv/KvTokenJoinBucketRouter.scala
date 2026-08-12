package briskkv

import chisel3._
import chisel3.util._

class RoutedKvTokenMetadata(
  metadataBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketIdBits: Int
) extends Bundle {
  val kZeroPoint = SInt(metadataBits.W)
  val kExponent = SInt(metadataBits.W)
  val vZeroPoint = SInt(metadataBits.W)
  val vExponent = SInt(metadataBits.W)
  val tokenTag = UInt(tagBits.W)
  val originalTokenIndex = UInt(tokenIndexBits.W)
  val routedTokenIndex = UInt(tokenIndexBits.W)
  val bucketId = UInt(bucketIdBits.W)
  val blockIndex = UInt(countBits.W)
  val last = Bool()
}

class RoutedKvFeatureValue(
  kBits: Int,
  vBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketIdBits: Int
) extends Bundle {
  val kQ = UInt(kBits.W)
  val vQ = UInt(vBits.W)
  val tokenTag = UInt(tagBits.W)
  val originalTokenIndex = UInt(tokenIndexBits.W)
  val routedTokenIndex = UInt(tokenIndexBits.W)
  val featureIndex = UInt(countBits.W)
  val bucketId = UInt(bucketIdBits.W)
  val blockIndex = UInt(countBits.W)
  val lastFeature = Bool()
  val last = Bool()
}

class KvTokenJoinBucketRouterStats extends Bundle {
  val activeCycles = UInt(64.W)
  val joinedTokens = UInt(64.W)
  val joinedValues = UInt(64.W)
  val routedTokens = UInt(64.W)
  val routedValues = UInt(64.W)
  val kWaitCycles = UInt(64.W)
  val vWaitCycles = UInt(64.W)
  val metadataStallCycles = UInt(64.W)
  val outputStallCycles = UInt(64.W)
  val rejectedBlocks = UInt(64.W)
}

class KvTokenJoinBucketRouterIO(
  kBits: Int,
  vBits: Int,
  metadataBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketIdBits: Int,
  bucketCountBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val blockIndex = Input(UInt(countBits.W))
  val blockLast = Input(Bool())
  val kMetadataIn = Flipped(
    Decoupled(new KvWriteQuantMetadata(metadataBits, tagBits))
  )
  val vMetadataIn = Flipped(
    Decoupled(new KvWriteQuantMetadata(metadataBits, tagBits))
  )
  val kQIn = Flipped(
    Decoupled(new KvWriteQuantizedValue(kBits, countBits, tagBits))
  )
  val vQIn = Flipped(
    Decoupled(new KvWriteQuantizedValue(vBits, countBits, tagBits))
  )
  val bucketCountsOut = Decoupled(
    new BucketCountRecord(bucketCountBits, countBits)
  )
  val metadataOut = Decoupled(
    new RoutedKvTokenMetadata(
      metadataBits,
      countBits,
      tagBits,
      tokenIndexBits,
      bucketIdBits
    )
  )
  val qOut = Decoupled(
    new RoutedKvFeatureValue(
      kBits,
      vBits,
      countBits,
      tagBits,
      tokenIndexBits,
      bucketIdBits
    )
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new KvTokenJoinBucketRouterStats)
}

/** Stable four-bucket K/V token router for the Format v0 write path.
  *
  * K and V metadata are accepted atomically, as are the corresponding q
  * features. Matching token tags, feature indices, and final markers are
  * checked before a complete 64-token block can produce output. The routing
  * score is the unsigned sum of all K q features for one token.
  *
  * After collecting a block, three equal-width integer thresholds classify
  * every token. Output scans original token indices separately for bucket
  * 0,1,2,3, which is equivalent to four stable FIFO queues and guarantees that
  * K q, V q, and both metadata records always share one permutation.
  */
class KvTokenJoinBucketRouter(
  maximumFeatureDim: Int = 256,
  metadataBits: Int = 8,
  countBits: Int = 32,
  tagBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  private val format = BriskKvFormatV0.params
  private val blockTokens = format.blockTokens
  private val bucketCount = format.bucketCount
  private val tokenIndexBits = log2Ceil(blockTokens)
  private val bucketIdBits = format.bucketIdBits
  private val bucketCountBits = format.bucketCountBits
  private val featureAddressBits = log2Ceil(maximumFeatureDim)
  private val featureCountBits = math.max(1, log2Ceil(maximumFeatureDim + 1))
  private val featureIndexBits = math.max(1, log2Ceil(maximumFeatureDim))
  private val memoryDepth = blockTokens * maximumFeatureDim
  private val scoreBits = format.kQuantBits + log2Ceil(maximumFeatureDim) + 1

  require(blockTokens == 64)
  require(bucketCount == 4)
  require(maximumFeatureDim >= 2 && isPow2(maximumFeatureDim))

  val io = IO(
    new KvTokenJoinBucketRouterIO(
      format.kQuantBits,
      format.vQuantBits,
      metadataBits,
      countBits,
      tagBits,
      tokenIndexBits,
      bucketIdBits,
      bucketCountBits
    )
  )

  private val Seq(
    sIdle,
    sMetadataIn,
    sValuesIn,
    sPrepareThresholds,
    sClassify,
    sCountClassified,
    sHeader,
    sFindToken,
    sMetadataOut,
    sValuesOut
  ) = Enum(10)
  val state = RegInit(sIdle)

  val kMemory = SyncReadMem(memoryDepth, UInt(format.kQuantBits.W))
  val vMemory = SyncReadMem(memoryDepth, UInt(format.vQuantBits.W))
  val kZeroPoints = Reg(Vec(blockTokens, SInt(metadataBits.W)))
  val kExponents = Reg(Vec(blockTokens, SInt(metadataBits.W)))
  val vZeroPoints = Reg(Vec(blockTokens, SInt(metadataBits.W)))
  val vExponents = Reg(Vec(blockTokens, SInt(metadataBits.W)))
  val tokenTags = Reg(Vec(blockTokens, UInt(tagBits.W)))
  val scores = Reg(Vec(blockTokens, UInt(scoreBits.W)))
  val bucketIds = Reg(Vec(blockTokens, UInt(bucketIdBits.W)))
  val bucketCounts = RegInit(VecInit(Seq.fill(bucketCount)(0.U(bucketCountBits.W))))

  val featureDimReg = RegInit(0.U(featureCountBits.W))
  val blockIndexReg = RegInit(0.U(countBits.W))
  val blockLastReg = RegInit(false.B)
  val collectTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val collectFeatureIndex = RegInit(0.U(featureIndexBits.W))
  val currentTokenTag = RegInit(0.U(tagBits.W))
  val scoreAccumulator = RegInit(0.U(scoreBits.W))
  val scoreMinimum = RegInit(0.U(scoreBits.W))
  val scoreMaximum = RegInit(0.U(scoreBits.W))
  val threshold1Reg = RegInit(0.U(scoreBits.W))
  val threshold2Reg = RegInit(0.U(scoreBits.W))
  val threshold3Reg = RegInit(0.U(scoreBits.W))
  val classifyIndex = RegInit(0.U(tokenIndexBits.W))
  val classifiedBucketReg = RegInit(0.U(bucketIdBits.W))
  val routeBucket = RegInit(0.U(bucketIdBits.W))
  val scanTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val selectedTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val routedTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val routeFeatureIndex = RegInit(0.U(featureIndexBits.W))
  val emittedInBucket = RegInit(0.U(bucketCountBits.W))
  val qOutputValid = RegInit(false.B)
  val qOutputReg = Reg(
    new RoutedKvFeatureValue(
      format.kQuantBits,
      format.vQuantBits,
      countBits,
      tagBits,
      tokenIndexBits,
      bucketIdBits
    )
  )
  val readOutstanding = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)

  val activeCycles = RegInit(0.U(64.W))
  val joinedTokens = RegInit(0.U(64.W))
  val joinedValues = RegInit(0.U(64.W))
  val routedTokens = RegInit(0.U(64.W))
  val routedValues = RegInit(0.U(64.W))
  val kWaitCycles = RegInit(0.U(64.W))
  val vWaitCycles = RegInit(0.U(64.W))
  val metadataStallCycles = RegInit(0.U(64.W))
  val outputStallCycles = RegInit(0.U(64.W))
  val rejectedBlocks = RegInit(0.U(64.W))

  io.busy := state =/= sIdle || qOutputValid || readOutstanding
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B

  val acceptingMetadata = state === sMetadataIn
  val metadataPairValid = io.kMetadataIn.valid && io.vMetadataIn.valid
  io.kMetadataIn.ready := acceptingMetadata && io.vMetadataIn.valid
  io.vMetadataIn.ready := acceptingMetadata && io.kMetadataIn.valid
  val metadataFire = acceptingMetadata && metadataPairValid

  val acceptingValues = state === sValuesIn
  val valuePairValid = io.kQIn.valid && io.vQIn.valid
  io.kQIn.ready := acceptingValues && io.vQIn.valid
  io.vQIn.ready := acceptingValues && io.kQIn.valid
  val valueFire = acceptingValues && valuePairValid

  io.bucketCountsOut.valid := state === sHeader
  io.bucketCountsOut.bits.counts := bucketCounts
  io.bucketCountsOut.bits.blockIndex := blockIndexReg
  io.bucketCountsOut.bits.last := blockLastReg

  io.metadataOut.valid := state === sMetadataOut
  io.metadataOut.bits.kZeroPoint := kZeroPoints(selectedTokenIndex)
  io.metadataOut.bits.kExponent := kExponents(selectedTokenIndex)
  io.metadataOut.bits.vZeroPoint := vZeroPoints(selectedTokenIndex)
  io.metadataOut.bits.vExponent := vExponents(selectedTokenIndex)
  io.metadataOut.bits.tokenTag := tokenTags(selectedTokenIndex)
  io.metadataOut.bits.originalTokenIndex := selectedTokenIndex
  io.metadataOut.bits.routedTokenIndex := routedTokenIndex
  io.metadataOut.bits.bucketId := routeBucket
  io.metadataOut.bits.blockIndex := blockIndexReg
  io.metadataOut.bits.last := routedTokenIndex === (blockTokens - 1).U

  io.qOut.valid := qOutputValid
  io.qOut.bits := qOutputReg

  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.joinedTokens := Mux(enableStats.B, joinedTokens, 0.U)
  io.stats.joinedValues := Mux(enableStats.B, joinedValues, 0.U)
  io.stats.routedTokens := Mux(enableStats.B, routedTokens, 0.U)
  io.stats.routedValues := Mux(enableStats.B, routedValues, 0.U)
  io.stats.kWaitCycles := Mux(enableStats.B, kWaitCycles, 0.U)
  io.stats.vWaitCycles := Mux(enableStats.B, vWaitCycles, 0.U)
  io.stats.metadataStallCycles := Mux(enableStats.B, metadataStallCycles, 0.U)
  io.stats.outputStallCycles := Mux(enableStats.B, outputStallCycles, 0.U)
  io.stats.rejectedBlocks := Mux(enableStats.B, rejectedBlocks, 0.U)

  val metadataFieldsValid =
    io.kMetadataIn.bits.zeroPoint >= (-(1 << (format.kZeroBits - 1))).S &&
      io.kMetadataIn.bits.zeroPoint <= ((1 << (format.kZeroBits - 1)) - 1).S &&
      io.vMetadataIn.bits.zeroPoint >= (-(1 << (format.vZeroBits - 1))).S &&
      io.vMetadataIn.bits.zeroPoint <= ((1 << (format.vZeroBits - 1)) - 1).S &&
      io.kMetadataIn.bits.exponent >= (-6).S &&
      io.kMetadataIn.bits.exponent <= 4.S &&
      io.vMetadataIn.bits.exponent >= (-6).S &&
      io.vMetadataIn.bits.exponent <= 4.S

  val inputAddress = Cat(
    collectTokenIndex,
    collectFeatureIndex(featureAddressBits - 1, 0)
  )
  val routeAddress = Cat(
    selectedTokenIndex,
    routeFeatureIndex(featureAddressBits - 1, 0)
  )
  val issueRead = state === sValuesOut && !qOutputValid && !readOutstanding
  val kReadValue = kMemory.read(routeAddress, issueRead)
  val vReadValue = vMemory.read(routeAddress, issueRead)
  val readResponseValid = RegNext(issueRead, false.B)

  when(readResponseValid) {
    qOutputReg.kQ := kReadValue
    qOutputReg.vQ := vReadValue
    qOutputReg.tokenTag := tokenTags(selectedTokenIndex)
    qOutputReg.originalTokenIndex := selectedTokenIndex
    qOutputReg.routedTokenIndex := routedTokenIndex
    qOutputReg.featureIndex := routeFeatureIndex
    qOutputReg.bucketId := routeBucket
    qOutputReg.blockIndex := blockIndexReg
    qOutputReg.lastFeature := routeFeatureIndex === featureDimReg - 1.U
    qOutputReg.last := routedTokenIndex === (blockTokens - 1).U &&
      routeFeatureIndex === featureDimReg - 1.U
    qOutputValid := true.B
    readOutstanding := false.B
  }
  when(issueRead) {
    readOutstanding := true.B
  }

  val span = scoreMaximum - scoreMinimum + 1.U
  val computedThreshold1 = scoreMinimum + ((span + 3.U) >> 2)
  val computedThreshold2 = scoreMinimum + (((span << 1) + 3.U) >> 2)
  val computedThreshold3 =
    scoreMinimum + ((span + (span << 1) + 3.U) >> 2)
  val scoreToClassify = scores(classifyIndex)
  val classifiedBucketWide =
    (scoreToClassify >= threshold1Reg) +&
      (scoreToClassify >= threshold2Reg) +&
      (scoreToClassify >= threshold3Reg)
  val classifiedBucket = classifiedBucketWide(bucketIdBits - 1, 0)

  val headerFire = io.bucketCountsOut.valid && io.bucketCountsOut.ready
  val metadataOutputFire = io.metadataOut.valid && io.metadataOut.ready
  val qOutputFire = io.qOut.valid && io.qOut.ready

  when(io.start && state === sIdle && !qOutputValid && !readOutstanding) {
    val commandValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U
    featureDimReg := io.featureDim(featureCountBits - 1, 0)
    blockIndexReg := io.blockIndex
    blockLastReg := io.blockLast
    collectTokenIndex := 0.U
    collectFeatureIndex := 0.U
    scoreAccumulator := 0.U
    threshold1Reg := 0.U
    threshold2Reg := 0.U
    threshold3Reg := 0.U
    classifyIndex := 0.U
    classifiedBucketReg := 0.U
    routeBucket := 0.U
    scanTokenIndex := 0.U
    routedTokenIndex := 0.U
    routeFeatureIndex := 0.U
    emittedInBucket := 0.U
    bucketCounts.foreach(_ := 0.U)
    qOutputValid := false.B
    readOutstanding := false.B
    errorReg := !commandValid
    doneReg := !commandValid
    activeCycles := 0.U
    joinedTokens := 0.U
    joinedValues := 0.U
    routedTokens := 0.U
    routedValues := 0.U
    kWaitCycles := 0.U
    vWaitCycles := 0.U
    metadataStallCycles := 0.U
    outputStallCycles := 0.U
    rejectedBlocks := Mux(commandValid, 0.U, 1.U)
    state := Mux(commandValid, sMetadataIn, sIdle)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle || qOutputValid || readOutstanding) {
      activeCycles := activeCycles + 1.U
    }
    when(
      (state === sMetadataIn && io.kMetadataIn.valid && !io.vMetadataIn.valid) ||
        (state === sValuesIn && io.kQIn.valid && !io.vQIn.valid)
    ) {
      vWaitCycles := vWaitCycles + 1.U
    }
    when(
      (state === sMetadataIn && io.vMetadataIn.valid && !io.kMetadataIn.valid) ||
        (state === sValuesIn && io.vQIn.valid && !io.kQIn.valid)
    ) {
      kWaitCycles := kWaitCycles + 1.U
    }
    when(io.metadataOut.valid && !io.metadataOut.ready) {
      metadataStallCycles := metadataStallCycles + 1.U
    }
    when(
      (io.bucketCountsOut.valid && !io.bucketCountsOut.ready) ||
        (io.qOut.valid && !io.qOut.ready)
    ) {
      outputStallCycles := outputStallCycles + 1.U
    }

    when(metadataFire) {
      when(
        io.kMetadataIn.bits.tokenTag =/= io.vMetadataIn.bits.tokenTag ||
          !metadataFieldsValid
      ) {
        errorReg := true.B
      }
      currentTokenTag := io.kMetadataIn.bits.tokenTag
      tokenTags(collectTokenIndex) := io.kMetadataIn.bits.tokenTag
      kZeroPoints(collectTokenIndex) := io.kMetadataIn.bits.zeroPoint
      kExponents(collectTokenIndex) := io.kMetadataIn.bits.exponent
      vZeroPoints(collectTokenIndex) := io.vMetadataIn.bits.zeroPoint
      vExponents(collectTokenIndex) := io.vMetadataIn.bits.exponent
      collectFeatureIndex := 0.U
      scoreAccumulator := 0.U
      state := sValuesIn
    }

    when(valueFire) {
      val expectedLast = collectFeatureIndex === featureDimReg - 1.U
      when(
        io.kQIn.bits.tokenTag =/= currentTokenTag ||
          io.vQIn.bits.tokenTag =/= currentTokenTag ||
          io.kQIn.bits.tokenTag =/= io.vQIn.bits.tokenTag ||
          io.kQIn.bits.featureIndex =/= collectFeatureIndex ||
          io.vQIn.bits.featureIndex =/= collectFeatureIndex ||
          io.kQIn.bits.featureIndex =/= io.vQIn.bits.featureIndex ||
          io.kQIn.bits.last =/= expectedLast ||
          io.vQIn.bits.last =/= expectedLast
      ) {
        errorReg := true.B
      }
      kMemory.write(inputAddress, io.kQIn.bits.q)
      vMemory.write(inputAddress, io.vQIn.bits.q)
      val completedScore = scoreAccumulator + io.kQIn.bits.q
      joinedValues := joinedValues + 1.U
      when(expectedLast) {
        scores(collectTokenIndex) := completedScore
        when(collectTokenIndex === 0.U) {
          scoreMinimum := completedScore
          scoreMaximum := completedScore
        }.otherwise {
          when(completedScore < scoreMinimum) {
            scoreMinimum := completedScore
          }
          when(completedScore > scoreMaximum) {
            scoreMaximum := completedScore
          }
        }
        joinedTokens := joinedTokens + 1.U
        when(collectTokenIndex === (blockTokens - 1).U) {
          classifyIndex := 0.U
          state := sPrepareThresholds
        }.otherwise {
          collectTokenIndex := collectTokenIndex + 1.U
          state := sMetadataIn
        }
      }.otherwise {
        scoreAccumulator := completedScore
        collectFeatureIndex := collectFeatureIndex + 1.U
      }
    }

    // Register the equal-width bucket thresholds before classification. This
    // breaks the score min/max -> threshold arithmetic -> comparisons path.
    when(state === sPrepareThresholds) {
      threshold1Reg := computedThreshold1(scoreBits - 1, 0)
      threshold2Reg := computedThreshold2(scoreBits - 1, 0)
      threshold3Reg := computedThreshold3(scoreBits - 1, 0)
      state := sClassify
    }

    // Classification and occupancy update intentionally occupy separate
    // cycles. The first cycle records the token bucket; the second performs a
    // fixed-register increment instead of a dynamic Vec read-modify-write.
    when(state === sClassify) {
      bucketIds(classifyIndex) := classifiedBucket
      classifiedBucketReg := classifiedBucket
      state := sCountClassified
    }

    when(state === sCountClassified) {
      switch(classifiedBucketReg) {
        is(0.U) { bucketCounts(0) := bucketCounts(0) + 1.U }
        is(1.U) { bucketCounts(1) := bucketCounts(1) + 1.U }
        is(2.U) { bucketCounts(2) := bucketCounts(2) + 1.U }
        is(3.U) { bucketCounts(3) := bucketCounts(3) + 1.U }
      }
      when(classifyIndex === (blockTokens - 1).U) {
        when(errorReg) {
          errorReg := true.B
          rejectedBlocks := rejectedBlocks + 1.U
          doneReg := true.B
          state := sIdle
        }.otherwise {
          state := sHeader
        }
      }.otherwise {
        classifyIndex := classifyIndex + 1.U
        state := sClassify
      }
    }

    when(headerFire) {
      routeBucket := 0.U
      scanTokenIndex := 0.U
      routedTokenIndex := 0.U
      emittedInBucket := 0.U
      state := sFindToken
    }

    when(state === sFindToken) {
      when(bucketCounts(routeBucket) === 0.U) {
        routeBucket := routeBucket + 1.U
        scanTokenIndex := 0.U
        emittedInBucket := 0.U
      }.elsewhen(bucketIds(scanTokenIndex) === routeBucket) {
        selectedTokenIndex := scanTokenIndex
        state := sMetadataOut
      }.otherwise {
        scanTokenIndex := scanTokenIndex + 1.U
      }
    }

    when(metadataOutputFire) {
      routeFeatureIndex := 0.U
      state := sValuesOut
    }

    when(qOutputFire) {
      qOutputValid := false.B
      routedValues := routedValues + 1.U
      when(qOutputReg.lastFeature) {
        val bucketFinished = emittedInBucket + 1.U === bucketCounts(routeBucket)
        routedTokens := routedTokens + 1.U
        routedTokenIndex := routedTokenIndex + 1.U
        emittedInBucket := emittedInBucket + 1.U
        when(qOutputReg.last) {
          doneReg := true.B
          state := sIdle
        }.elsewhen(bucketFinished) {
          routeBucket := routeBucket + 1.U
          scanTokenIndex := 0.U
          emittedInBucket := 0.U
          state := sFindToken
        }.otherwise {
          scanTokenIndex := selectedTokenIndex + 1.U
          state := sFindToken
        }
      }.otherwise {
        routeFeatureIndex := routeFeatureIndex + 1.U
      }
    }
  }
}
