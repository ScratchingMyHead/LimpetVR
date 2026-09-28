/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Persistent ring log: logcat rotates in minutes under system spam, which
 *  keeps eating our freeze forensics. This mirrors critical events to
 *  filesDir/limpetvr.log (capped ~256KB, oldest dropped first) so a repro can
 *  be pulled ANY time after with: adb exec-out run-as net.limpetvr.player cat files/limpetvr.log
 */
object FileLog {
    private const val TAG = "LimpetVR"
    private const val NAME = "limpetvr.log"
    private const val CAP = 256 * 1024L
    private val lock = Any()
    private var dir: File? = null
    private val t0 = System.currentTimeMillis()
    private val fmt = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    fun init(ctx: Context) {
        dir = ctx.filesDir
        i("FileLog", "session start")
    }

    private fun file(): File? = dir?.let { File(it, NAME) }

    fun i(tag: String, msg: String) {
        Log.i(tag, msg)
        write("I/$tag: $msg")
    }

    fun d(tag: String, msg: String) {
        Log.d(tag, msg)
        write("D/$tag: $msg")
    }

    fun w(tag: String, msg: String) {
        Log.w(tag, msg)
        write("W/$tag: $msg")
    }

    fun e(tag: String, msg: String, tr: Throwable? = null) {
        if (tr != null) Log.e(tag, msg, tr) else Log.e(tag, msg)
        write("E/$tag: $msg ${tr?.message ?: ""}")
    }

    private fun write(line: String) {
        val f = file() ?: return
        synchronized(lock) {
            try {
                if (f.exists() && f.length() > CAP) {
                    // drop oldest half
                    val bytes = f.readBytes()
                    val half = bytes.size / 2
                    var cut = half
                    while (cut < bytes.size && bytes[cut] != '\n'.code.toByte()) cut++
                    f.writeBytes(bytes.copyOfRange((cut + 1).coerceAtMost(bytes.size), bytes.size))
                }
                f.appendText("${fmt.format(Date())} $line\n")
            } catch (_: Exception) {}
        }
    }
}
