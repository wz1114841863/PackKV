package briskkv

import java.nio.charset.StandardCharsets
import java.nio.{ByteBuffer, ByteOrder}
import java.security.MessageDigest
import java.util.regex.Pattern

object GoldenVectorLoader {
  private val root = "/golden_vectors"

  private def readResource(path: String): Array[Byte] = {
    val stream = Option(getClass.getResourceAsStream(path)).getOrElse {
      throw new IllegalArgumentException(s"Missing test resource: $path")
    }
    try stream.readAllBytes()
    finally stream.close()
  }

  private def descriptor(caseName: String): String =
    new String(
      readResource(s"$root/$caseName/descriptor.json"),
      StandardCharsets.UTF_8
    )

  private def bitpackSection(caseName: String, cache: String): String = {
    require(cache == "k" || cache == "v")
    val text = descriptor(caseName)
    val bitpackStart = text.indexOf("\"bitpack\"")
    val bucketStart = text.indexOf("\"bucket\"", bitpackStart)
    require(bitpackStart >= 0 && bucketStart > bitpackStart)
    val bitpack = text.substring(bitpackStart, bucketStart)
    val pattern = (s"(?s)\"$cache\"\\s*:\\s*\\{(.*?)\\}").r
    pattern
      .findFirstMatchIn(bitpack)
      .map(_.group(1))
      .getOrElse(throw new IllegalArgumentException(s"bitpack.$cache missing"))
  }

  def bitpackInt(caseName: String, cache: String, key: String): Int = {
    val pattern = (s"\"${Pattern.quote(key)}\"\\s*:\\s*(\\d+)").r
    pattern
      .findFirstMatchIn(bitpackSection(caseName, cache))
      .map(_.group(1).toInt)
      .getOrElse(throw new IllegalArgumentException(s"bitpack.$cache.$key missing"))
  }

  def bitpackBoolean(caseName: String, cache: String, key: String): Boolean = {
    val pattern = (s"\"${Pattern.quote(key)}\"\\s*:\\s*(true|false)").r
    pattern
      .findFirstMatchIn(bitpackSection(caseName, cache))
      .map(_.group(1).toBoolean)
      .getOrElse(throw new IllegalArgumentException(s"bitpack.$cache.$key missing"))
  }

  private def expectedMetadata(
    caseName: String,
    fileName: String
  ): (Int, String) = {
    val quoted = Pattern.quote(fileName)
    val blockPattern = (s"(?s)\"$quoted\"\\s*:\\s*\\{(.*?)\\}").r
    val block = blockPattern
      .findFirstMatchIn(descriptor(caseName))
      .map(_.group(1))
      .getOrElse(throw new IllegalArgumentException(s"$fileName missing from descriptor"))
    val byteCount = "\"bytes\"\\s*:\\s*(\\d+)".r
      .findFirstMatchIn(block)
      .map(_.group(1).toInt)
      .getOrElse(throw new IllegalArgumentException(s"bytes missing for $fileName"))
    val checksum = "\"sha256\"\\s*:\\s*\"([0-9a-f]{64})\"".r
      .findFirstMatchIn(block)
      .map(_.group(1))
      .getOrElse(throw new IllegalArgumentException(s"sha256 missing for $fileName"))
    (byteCount, checksum)
  }

  private def sha256(bytes: Array[Byte]): String =
    MessageDigest
      .getInstance("SHA-256")
      .digest(bytes)
      .map(byte => f"${byte & 0xff}%02x")
      .mkString

  def bytes(caseName: String, fileName: String): Array[Byte] = {
    val data = readResource(s"$root/$caseName/$fileName")
    val (expectedBytes, expectedSha256) = expectedMetadata(caseName, fileName)
    require(data.length == expectedBytes, s"$fileName byte length mismatch")
    require(sha256(data) == expectedSha256, s"$fileName SHA-256 mismatch")
    data
  }

  def unsigned(caseName: String, fileName: String): IndexedSeq[Int] =
    bytes(caseName, fileName).map(_ & 0xff).toIndexedSeq

  def signed(caseName: String, fileName: String): IndexedSeq[Int] =
    bytes(caseName, fileName).map(_.toInt).toIndexedSeq

  def float32LittleEndian(
    caseName: String,
    fileName: String
  ): IndexedSeq[Float] = {
    val data = bytes(caseName, fileName)
    require(data.length % 4 == 0)
    val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
    IndexedSeq.fill(data.length / 4)(buffer.getFloat())
  }

  def unpackFixed(
    data: IndexedSeq[Int],
    fieldBits: Int,
    fieldCount: Int,
    signed: Boolean
  ): IndexedSeq[Int] = {
    require(fieldBits >= 1 && fieldBits <= 8)
    require(data.length == (fieldBits * fieldCount + 7) / 8)
    val raw = data.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (byte, index)) =>
      acc | (BigInt(byte & 0xff) << (8 * index))
    }
    require((raw >> (fieldBits * fieldCount)) == 0, "non-zero padding bits")
    val mask = (BigInt(1) << fieldBits) - 1
    (0 until fieldCount).map { index =>
      val unsigned = ((raw >> (index * fieldBits)) & mask).toInt
      if (signed && (unsigned & (1 << (fieldBits - 1))) != 0)
        unsigned - (1 << fieldBits)
      else unsigned
    }
  }
}
