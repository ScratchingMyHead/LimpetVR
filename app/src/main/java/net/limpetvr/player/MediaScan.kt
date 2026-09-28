/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.content.Context
import android.media.MediaExtractor
import android.net.Uri
import java.nio.ByteBuffer

/** File-structure probe: walks per-track sample timestamps (headers only,
 *  no decode) to find short tracks, gaps and discontinuities — the prime
 *  suspects when video freezes with full buffers and no errors. Works on
 *  local files and on SMB files via a temp proxy URL (no download).
 */
object MediaScan {
    data class TrackInfo(
        val index: Int,
        val mime: String,
        val label: String,
        val samples: Int,
        val firstUs: Long,
        val lastUs: Long,
        val maxGapUs: Long,
        val bytes: Long,
        val truncated: Boolean
    )
    data class Result(val uri: String, val tracks: List<TrackInfo>, val error: String?)

    private const val MAX_SAMPLES = 100_000
    // Stop a track once it demonstrably covers past any freeze point.
    private const val COVER_US = 120_000_000L

    fun scan(ctx: Context, uri: Uri, label: String): Result {
        val ex = MediaExtractor()
        try {
            ex.setDataSource(ctx, uri, null)
        } catch (e: Exception) {
            return Result(label, emptyList(), "open failed: ${e.message}")
        }
        val out = mutableListOf<TrackInfo>()
        try {
            val buf = ByteBuffer.allocate(4 * 1024 * 1024)
            for (ti in 0 until ex.trackCount) {
                val fmt = try { ex.getTrackFormat(ti) } catch (e: Exception) { continue }
                val mime = fmt.getString("mime") ?: "?"
                val isVideo = mime.startsWith("video/")
                val isAudio = mime.startsWith("audio/")
                if (!isVideo && !isAudio) continue
                val extra = when {
                    isVideo -> " ${fmt.getInteger("width", -1)}x${fmt.getInteger("height", -1)}"
                    else -> ""
                }
                ex.selectTrack(ti)
                var n = 0; var first = -1L; var last = -1L; var maxGap = 0L
                var prev = -1L; var bytes = 0L; var truncated = false
                try {
                    while (n < MAX_SAMPLES) {
                        val t = ex.sampleTime
                        if (t < 0) break
                        if (first < 0) first = t
                        if (prev >= 0 && t > prev) maxGap = maxGap.coerceAtLeast(t - prev)
                        prev = t; last = t
                        if (last > COVER_US && n > 100) break // covered past any freeze
                        val sz = try {
                            ex.readSampleData(buf, 0)
                        } catch (e: Exception) {
                            break
                        }
                        if (sz < 0) break
                        bytes += sz
                        n++
                        if (!ex.advance()) break
                    }
                    if (n >= MAX_SAMPLES) truncated = true
                } catch (e: Exception) {
                    truncated = true
                } finally {
                    try { ex.unselectTrack(ti) } catch (_: Exception) {}
                }
                out += TrackInfo(ti, mime, "$mime$extra", n, first, last, maxGap, bytes, truncated)
            }
        } finally {
            try { ex.release() } catch (_: Exception) {}
        }
        if (out.isEmpty()) return Result(label, out, "no A/V tracks found")
        return Result(label, out, null)
    }

    fun formatMs(us: Long): String {
        if (us < 0) return "?"
        val s = us / 1_000_000
        return "${s / 60}:${(s % 60).toString().padStart(2, '0')}"
    }

    fun summarize(r: Result): String {
        val sb = StringBuilder(r.uri.takeLast(40)).append("\n\n")
        r.error?.let { sb.append("ERROR: $it\n"); return sb.toString() }
        for (t in r.tracks) {
            sb.append("${t.label}\n")
            sb.append("  samples=${t.samples}${if (t.truncated) "+" else ""}  " +
                "span=${formatMs(t.firstUs)}→${formatMs(t.lastUs)}  " +
                "maxGap=${t.maxGapUs / 1000}ms\n")
        }
        // verdict line: video coverage vs audio coverage
        val v = r.tracks.filter { it.mime.startsWith("video/") }
        val a = r.tracks.filter { it.mime.startsWith("audio/") }
        if (v.isNotEmpty() && a.isNotEmpty()) {
            val vl = v.maxOf { it.lastUs }; val al = a.maxOf { it.lastUs }
            sb.append("\nvideo covers ${formatMs(vl)}, audio covers ${formatMs(al)}")
            if (vl >= 0 && al - vl > 5_000_000L) {
                sb.append("\n⚠ VIDEO ENDS ${(al - vl) / 1_000_000}s BEFORE AUDIO — freeze explained")
            }
            val vg = v.maxOf { it.maxGapUs }
            if (vg > 1_000_000L) {
                sb.append("\n⚠ video gap ${vg / 1000}ms — freeze explained")
            }
        }
        return sb.toString()
    }
}
