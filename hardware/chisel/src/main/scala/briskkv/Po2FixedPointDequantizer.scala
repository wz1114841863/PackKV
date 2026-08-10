package briskkv

import chisel3._
import chisel3.util._

class Po2DequantInput(codeValueBits: Int, metadataBits: Int) extends Bundle {
  val q = UInt(codeValueBits.W)
  val zeroPoint = SInt(metadataBits.W)
  val exponent = SInt(metadataBits.W)
}

class Po2DequantOutput(outputBits: Int) extends Bundle {
  val fixedRaw = SInt(outputBits.W)
  val error = Bool()
}

class Po2FixedPointDequantizerIO(
  codeValueBits: Int,
  metadataBits: Int,
  outputBits: Int
) extends Bundle {
  val in = Flipped(Decoupled(new Po2DequantInput(codeValueBits, metadataBits)))
  val out = Decoupled(new Po2DequantOutput(outputBits))
}

/** Exact multiplier-free dequantization into a common fixed-point format.
  *
  * For Format v0, fractionalBits=6 and exponent is restricted to [-6, 4]:
  *   fixedRaw = (q + zeroPoint) << (exponent + fractionalBits)
  *   realValue = fixedRaw / 2^fractionalBits
  */
class Po2FixedPointDequantizer(
  codeValueBits: Int,
  zeroPointBits: Int,
  fractionalBits: Int = 6,
  minimumExponent: Int = -6,
  maximumExponent: Int = 4,
  metadataBits: Int = 8,
  outputBits: Int = 18
) extends Module {
  require(codeValueBits >= 1 && codeValueBits <= 8)
  require(zeroPointBits >= 2 && zeroPointBits <= metadataBits)
  require(fractionalBits >= -minimumExponent)
  require(minimumExponent <= maximumExponent)
  private val mantissaBits = math.max(codeValueBits + 1, zeroPointBits) + 1
  require(outputBits >= mantissaBits + maximumExponent + fractionalBits)

  val io = IO(
    new Po2FixedPointDequantizerIO(codeValueBits, metadataBits, outputBits)
  )

  val qSigned = Cat(0.U(1.W), io.in.bits.q).asSInt
  val mantissa = qSigned +& io.in.bits.zeroPoint
  val mantissaWide = Wire(SInt(outputBits.W))
  mantissaWide := mantissa

  private val maximumShift = maximumExponent + fractionalBits
  val exponentValid = io.in.bits.exponent >= minimumExponent.S &&
    io.in.bits.exponent <= maximumExponent.S
  val zeroPointValid =
    io.in.bits.zeroPoint >= (-(1 << (zeroPointBits - 1))).S &&
      io.in.bits.zeroPoint <= ((1 << (zeroPointBits - 1)) - 1).S
  val shiftAmount = (io.in.bits.exponent + fractionalBits.S).asUInt
  val shifted = MuxLookup(
    shiftAmount,
    0.S(outputBits.W)
  )(
    (0 to maximumShift).map { amount =>
      amount.U -> (mantissaWide << amount)(outputBits - 1, 0).asSInt
    }
  )

  io.in.ready := io.out.ready
  io.out.valid := io.in.valid
  io.out.bits.fixedRaw := shifted
  io.out.bits.error := !exponentValid || !zeroPointValid
}
