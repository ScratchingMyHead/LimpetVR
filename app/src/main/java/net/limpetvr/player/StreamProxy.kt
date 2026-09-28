/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.util.Log
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlin.concurrent.thread

/**
 * Localhost HTTP proxy: turns SMB files into seekable
 * http://127.0.0.1:port/v/<token>/file URLs for ExoPlayer.
 * Supports GET/HEAD + single `Range: bytes=` so playback starts instantly
 * and seeks without ever downloading the whole file. Nothing is cached to disk.
 *
 * Transfer model: every HTTP response opens its OWN SMB handle and streams
 * its own byte range sequentially, then closes it. No shared buffers, no
 * generations, no cross-request state — concurrent/overlapping ranges (moov
 * re-reads, retries, seeks) proceed independently like any HTTP file server.
 * Transient SMB hiccups are retried transparently inside the response.
 */
object StreamProxy {
    private const val TAG = "StreamProxy"
    private const val WINDOW = 512 * 1024
    private const val MAX_TOKENS = 16
    private const val READ_RETRIES = 5
    private const val REOPEN_LIMIT = 3
    private var server: ServerSocket? = null
    private var acceptThread: Thread? = null
    private val pool = Executors.newFixedThreadPool(8)
    private val sources = ConcurrentHashMap<String, TokenStream>()

    class TokenStream(
        val opener: () -> SmbReadHandle,
        val size: Long,
        val mime: String,
        val displayName: String
    ) {
        // live GET responses currently streaming this token (diagnostics)
        @Volatile var activeResponses = 0
    }

    @Synchronized
    fun start(): Int {
        if (server != null) return server!!.localPort
        val s = ServerSocket(0, 50, java.net.InetAddress.getByName("127.0.0.1"))
        server = s
        acceptThread = thread(name = "limpetvr-proxy", isDaemon = true) {
            while (!s.isClosed) {
                try {
                    val sock = s.accept()
                    pool.execute { handleSocket(sock) }
                } catch (e: Exception) {
                    if (!s.isClosed) Log.w(TAG, "accept: ${e.message}")
                }
            }
        }
        return s.localPort
    }

    @Synchronized
    fun stop() {
        try { server?.close() } catch (_: Exception) {}
        server = null
    }

    fun baseUrl(): String = "http://127.0.0.1:${server?.localPort ?: start()}"

    /** Register a stream, get a playback URL. Size probed once by the caller. */
    fun register(opener: () -> SmbReadHandle, size: Long, displayName: String): String {
        while (sources.size >= MAX_TOKENS) {
            val it = sources.keys.iterator()
            if (!it.hasNext()) break
            // Evict from the MAP only: in-flight responses hold their own
            // handles and finish undisturbed; only new requests are refused.
            sources.remove(it.next())
        }
        val token = UUID.randomUUID().toString().replace("-", "")
        sources[token] = TokenStream(opener, size, mimeFor(displayName), displayName)
        FileLog.d(TAG, "register ${token.take(8)} size=$size name=$displayName")
        return "${baseUrl()}/v/$token/${displayName.encodeForUrl()}"
    }

    fun unregisterByUrl(url: String) {
        val token = url.substringAfter("/v/", "").substringBefore("/")
        sources.remove(token)
    }

    fun mimeFor(name: String): String {
        val n = name.lowercase()
        return when {
            n.endsWith(".mkv") -> "video/x-matroska"
            n.endsWith(".avi") -> "video/x-msvideo"
            n.endsWith(".mov") -> "video/quicktime"
            n.endsWith(".webm") -> "video/webm"
            n.endsWith(".ts") -> "video/mp2t"
            n.endsWith(".mpg") || n.endsWith(".mpeg") -> "video/mpeg"
            else -> "video/mp4"
        }
    }

    private fun String.encodeForUrl(): String =
        java.net.URLEncoder.encode(this, "UTF-8").replace("+", "%20")

    private fun handleSocket(sock: Socket) {
        sock.use { s ->
            s.soTimeout = 15000
            try {
                val input = BufferedInputStream(s.getInputStream())
                val req = readHeaders(input) ?: return
                val out = BufferedOutputStream(s.getOutputStream())
                serve(req, out)
                out.flush()
            } catch (e: Exception) {
                Log.w(TAG, "serve: ${e.message}")
            }
        }
    }

    private data class Req(val method: String, val path: String, val headers: Map<String, String>)

    private fun readHeaders(input: BufferedInputStream): Req? {
        val head = ByteArrayOutputStream()
        val one = ByteArray(1)
        while (head.size() < 32768) {
            val n = input.read(one)
            if (n <= 0) return null
            head.write(one[0].toInt())
            val b = head.toByteArray()
            val l = b.size
            if (l >= 4 && b[l - 4] == 13.toByte() && b[l - 3] == 10.toByte() &&
                b[l - 2] == 13.toByte() && b[l - 1] == 10.toByte()
            ) break
        }
        val text = head.toString("ISO-8859-1")
        val lines = text.split("\r\n")
        if (lines.isEmpty()) return null
        val parts = lines[0].split(" ")
        if (parts.size < 2) return null
        val headers = mutableMapOf<String, String>()
        for (i in 1 until lines.size) {
            val ln = lines[i]
            if (ln.isBlank()) break
            val idx = ln.indexOf(':')
            if (idx > 0) headers[ln.substring(0, idx).trim().lowercase()] = ln.substring(idx + 1).trim()
        }
        return Req(parts[0].uppercase(), parts[1], headers)
    }

    private fun serve(req: Req, out: BufferedOutputStream) {
        val rawPath = req.path.substringBefore("?")
        val segs = rawPath.split("/")
        if (segs.size < 3 || segs[1] != "v") {
            writeStatus(out, 404, "Not Found", emptyMap()); return
        }
        val token = URLDecoder.decode(segs[2], "UTF-8")
        val src = sources[token]
        if (src == null) {
            FileLog.w(TAG, "${req.method} ${rawPath.take(40)} -> 404 unknown token")
            writeStatus(out, 404, "Not Found", emptyMap()); return
        }
        val size = src.size
        val rangeHeader = req.headers["range"]
        var start = 0L; var end = size - 1; var partial = false
        if (rangeHeader != null && rangeHeader.startsWith("bytes=")) {
            val spec = rangeHeader.removePrefix("bytes=").split(",")[0].trim()
            val dash = spec.indexOf('-')
            if (dash >= 0) {
                val a = spec.substring(0, dash); val b = spec.substring(dash + 1)
                try {
                    if (a.isEmpty()) {
                        val n = b.toLong()
                        start = (size - n).coerceAtLeast(0); end = size - 1
                    } else {
                        start = a.toLong()
                        end = if (b.isEmpty()) size - 1 else b.toLong().coerceAtMost(size - 1)
                    }
                    partial = start <= end && start < size
                } catch (_: Exception) {}
            }
        }
        if (req.method == "HEAD") {
            writeStatus(out, 200, "OK", mapOf(
                "Content-Type" to src.mime,
                "Content-Length" to size.toString(),
                "Accept-Ranges" to "bytes"
            ))
            return
        }
        if (rangeHeader != null && !partial) {
            writeStatus(out, 416, "Range Not Satisfiable",
                mapOf("Content-Range" to "bytes */$size"))
            return
        }
        val length = end - start + 1
        val t = token.take(8)
        val concurrent: Int
        synchronized(src) {
            src.activeResponses++
            concurrent = src.activeResponses
        }
        Log.d(TAG, "${req.method} $t range=$rangeHeader -> ${if (partial) 206 else 200} [$start-$end/$size]" +
            if (concurrent > 1) " CONCURRENT=$concurrent" else "")
        if (req.method != "GET") {
            synchronized(src) { src.activeResponses-- }
            writeStatus(out, if (partial) 206 else 200, if (partial) "Partial Content" else "OK",
                mapOf(
                    "Content-Type" to src.mime,
                    "Accept-Ranges" to "bytes",
                    "Content-Length" to length.toString(),
                    "Connection" to "close"
                ) + if (partial) mapOf("Content-Range" to "bytes $start-$end/$size") else emptyMap())
            return
        }
        // Open FIRST so open-failures surface as HTTP 500 (retryable) rather
        // than a silently truncated 200.
        val handle: SmbReadHandle = try {
            src.opener()
        } catch (e: Exception) {
            FileLog.w(TAG, "$t open failed: ${e.message}")
            synchronized(src) { src.activeResponses-- }
            writeStatus(out, 500, "Internal Server Error", emptyMap())
            return
        }
        val headers = mutableMapOf(
            "Content-Type" to src.mime,
            "Accept-Ranges" to "bytes",
            "Content-Length" to length.toString(),
            "Connection" to "close"
        )
        if (partial) headers["Content-Range"] = "bytes $start-$end/$size"
        writeStatus(out, if (partial) 206 else 200, if (partial) "Partial Content" else "OK", headers)
        serveRange(src, token, handle, start, end, out)
    }

    /** Stream [start, end] through this response's OWN handle. Seeks, retries
     *  and moov re-reads open independent responses; nothing is shared, so
     *  overlapping ranges can never corrupt each other. Transient SMB
     *  hiccups are retried transparently inside the response. */
    private fun serveRange(
        src: TokenStream, token: String, handle: SmbReadHandle,
        start: Long, end: Long, out: BufferedOutputStream
    ) {
        val t = token.take(8)
        val t0 = System.currentTimeMillis()
        var pos = start
        var served = 0L
        val buf = ByteArray(WINDOW)
        // The session can die while the app sits in the background (server
        // keepalive/NAT timeout), which kills this handle for good — then only
        // a fresh open (SmbManager re-authenticates) can continue the range.
        var h = handle
        var reopens = 0
        try {
            while (pos <= end) {
                val want = minOf(buf.size.toLong(), end - pos + 1).toInt()
                var got = 0
                var failures = 0
                while (got < want) {
                    val n = try {
                        h.readAt(pos + got, buf, got, want - got)
                    } catch (e: Exception) {
                        failures++
                        if (failures > READ_RETRIES) {
                            FileLog.w(TAG, "$t read @$pos failed ${failures}x, aborting: ${e.message}")
                            return
                        }
                        if (reopens < REOPEN_LIMIT) {
                            try { h.close() } catch (_: Exception) {}
                            try {
                                h = src.opener(); reopens++
                                Log.d(TAG, "$t re-opened SMB handle @$pos (reopen $reopens) after: ${e.message}")
                            } catch (o: Exception) {
                                Log.d(TAG, "$t reopen @$pos failed: ${o.message}")
                            }
                        }
                        Log.d(TAG, "$t read @$pos hiccup (${e.message}), retry $failures")
                        try { Thread.sleep(300) } catch (_: InterruptedException) { return }
                        continue
                    }
                    if (n <= 0) break // EOF (short file vs probed size)
                    got += n
                    failures = 0
                }
                if (got <= 0) break
                try {
                    out.write(buf, 0, got)
                    out.flush()
                } catch (_: Exception) {
                    return // client went away (seek/close) — normal
                }
                pos += got; served += got
            }
            if (pos <= end) {
                FileLog.w(TAG, "$t ENDED EARLY at $pos (asked $start-$end)")
            }
        } finally {
            val dt = (System.currentTimeMillis() - t0).coerceAtLeast(1)
            try { h.close() } catch (_: Exception) {}
            synchronized(src) { src.activeResponses-- }
            FileLog.d(TAG, "served $t $served B in ${dt}ms (${served * 1000 / dt / 1024} KB/s)")
        }
    }

    private fun writeStatus(out: BufferedOutputStream, code: Int, msg: String, headers: Map<String, String>) {
        val sb = StringBuilder()
        sb.append("HTTP/1.1 ").append(code).append(' ').append(msg).append("\r\n")
        for ((k, v) in headers) sb.append(k).append(": ").append(v).append("\r\n")
        sb.append("\r\n")
        out.write(sb.toString().toByteArray(Charsets.ISO_8859_1))
    }
}
