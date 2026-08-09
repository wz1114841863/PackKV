package briskkv

import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class BriskKvFormatSpec extends AnyFreeSpec with Matchers {
  private val p = BriskKvFormatV0.params

  "Format v0 must preserve the qualified field limits" in {
    p.kQuantBits mustBe 6
    p.vQuantBits mustBe 4
    p.kZeroBits mustBe 7
    p.vZeroBits mustBe 5
    p.exponentBits mustBe 4
    p.packWidthBits mustBe 3

    -(1 << (p.kZeroBits - 1)) must be <= -34
    (1 << p.kQuantBits) - 1 must be >= 48
    -(1 << (p.exponentBits - 1)) must be <= -6
    (1 << (p.exponentBits - 1)) - 1 must be >= 4
  }

  "Format v0 must use a three-byte bucket header per 64-token block" in {
    p.bucketIdBits mustBe 2
    p.bucketCountBits mustBe 7
    p.storedBucketCounts mustBe 3
    p.bucketHeaderBitsPerBlock mustBe 21
    p.bucketHeaderBytesPerBlock mustBe 3
  }

  "compact quantization metadata must reduce 64 bits to 20 bits" in {
    p.nativeMetadataBitsPerKvParameter mustBe 64
    p.compactMetadataBitsPerKvParameter mustBe 20
    val reduction = 1.0 - p.compactMetadataBitsPerKvParameter.toDouble / p.nativeMetadataBitsPerKvParameter
    reduction mustBe 0.6875
  }
}
