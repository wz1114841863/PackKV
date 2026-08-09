package briskkv

/** Frozen field widths and structural constants for BRISK-KV Format v0.
  *
  * These values are qualified by the current three-model, three-sample trace
  * set. They are hardware-format limits, not mathematical bounds for arbitrary
  * tensors. Encoders must reject values that do not fit.
  */
final case class BriskKvFormatParams(
  blockTokens: Int,
  packTokens: Int,
  bucketCount: Int,
  kQuantBits: Int,
  vQuantBits: Int,
  kZeroBits: Int,
  vZeroBits: Int,
  exponentBits: Int,
  packWidthBits: Int
) {
  require(blockTokens > 0)
  require(packTokens > 0 && blockTokens % packTokens == 0)
  require(bucketCount >= 2 && (bucketCount & (bucketCount - 1)) == 0)
  require(Seq(kQuantBits, vQuantBits, kZeroBits, vZeroBits, exponentBits, packWidthBits).forall(_ > 0))

  val bucketIdBits: Int = Integer.numberOfTrailingZeros(bucketCount)
  val bucketCountBits: Int = 32 - Integer.numberOfLeadingZeros(blockTokens)
  val storedBucketCounts: Int = bucketCount - 1
  val bucketHeaderBitsPerBlock: Int = storedBucketCounts * bucketCountBits
  val bucketHeaderBytesPerBlock: Int = (bucketHeaderBitsPerBlock + 7) / 8
  val compactMetadataBitsPerKvParameter: Int = kZeroBits + exponentBits + vZeroBits + exponentBits
  val nativeMetadataBitsPerKvParameter: Int = 64
}

object BriskKvFormatV0 {
  val params: BriskKvFormatParams = BriskKvFormatParams(
    blockTokens = 64,
    packTokens = 16,
    bucketCount = 4,
    kQuantBits = 6,
    vQuantBits = 4,
    kZeroBits = 7,
    vZeroBits = 5,
    exponentBits = 4,
    packWidthBits = 3
  )
}
