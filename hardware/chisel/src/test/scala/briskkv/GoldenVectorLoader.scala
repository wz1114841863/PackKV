package briskkv

import java.nio.charset.StandardCharsets
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
}
