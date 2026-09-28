/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import com.hierynomus.msdtyp.AccessMask
import com.hierynomus.mssmb2.SMB2CreateDisposition
import com.hierynomus.mssmb2.SMB2ShareAccess
import com.hierynomus.mssmb2.SMBApiException
import com.hierynomus.smbj.SMBClient
import com.hierynomus.smbj.auth.AuthenticationContext
import com.hierynomus.smbj.session.Session
import com.hierynomus.smbj.share.DiskShare
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.EnumSet

private const val FILE_ATTRIBUTE_DIRECTORY = 0x10
private const val TAG = "LimpetVR"

// NtStatus answers that mean the session itself is gone, not the path.
private const val STATUS_USER_SESSION_DELETED = 0xC0000203L

/**
 * SMB2/3 client over smbj. All blocking I/O on Dispatchers.IO.
 * Streaming is random-access via [SmbReadHandle] — the proxy seeks with
 * readAt() so playback never downloads the whole file.
 */
class SmbManager {
    private var client: SMBClient? = null
    private var session: Session? = null
    private var share: DiskShare? = null
    private var boundConnId: String = ""
    // Credentials of the bound connection. A backgrounded app keeps no live
    // socket: the server/NAT drops it after a while, so every operation must
    // be able to re-authenticate on its own instead of failing forever.
    private var boundConn: SmbConnection? = null
    private val reconnectLock = Mutex()

    @Volatile var connected: Boolean = false
        private set

    suspend fun connect(c: SmbConnection) = withContext(Dispatchers.IO) {
        disconnectLocked()
        val cl = SMBClient()
        val con = cl.connect(c.host)
        val auth = if (c.username.isBlank()) AuthenticationContext.anonymous()
            else AuthenticationContext(c.username, c.password.toCharArray(), c.domain)
        val sess = con.authenticate(auth)
        val disk = sess.connectShare(c.share) as DiskShare
        client = cl; session = sess; share = disk
        boundConnId = c.id; boundConn = c; connected = true
    }

    suspend fun list(path: String): List<SmbEntry> = withContext(Dispatchers.IO) {
        withReconnect("list") { sh ->
            val dir = path.ifBlank { "" }
            sh.list(dir).mapNotNull { info ->
                val nm = info.fileName
                if (nm == "." || nm == "..") return@mapNotNull null
                val isDir = (info.fileAttributes and FILE_ATTRIBUTE_DIRECTORY.toLong()) != 0L
                val rel = if (dir.isBlank()) nm else "$dir\\$nm"
                SmbEntry(nm, rel, isDir, info.endOfFile)
            }.sortedWith(compareBy({ !it.isDir }, { it.name.lowercase() }))
        }
    }

    /** Open a random-access handle. Caller MUST close. */
    suspend fun openRead(path: String): SmbReadHandle = withContext(Dispatchers.IO) {
        withReconnect("openRead") { sh ->
            val f = sh.openFile(
                path,
                EnumSet.of(AccessMask.GENERIC_READ),
                null,
                EnumSet.of(SMB2ShareAccess.FILE_SHARE_READ, SMB2ShareAccess.FILE_SHARE_WRITE, SMB2ShareAccess.FILE_SHARE_DELETE),
                SMB2CreateDisposition.FILE_OPEN,
                null
            )
            val size = f.fileInformation.standardInformation.endOfFile
            SmbReadHandle(f, size)
        }
    }

    /**
     * Run [block] on the bound share. After a long background the session is
     * silently gone while [connected] still reads true: re-authenticate once
     * and retry. Protocol-level answers (missing path, access denied) mean the
     * server answered, so they propagate untouched.
     */
    private suspend fun <T> withReconnect(what: String, block: (DiskShare) -> T): T {
        val sh = share
        if (sh == null) {
            val c = boundConn ?: error("not connected")
            FileLog.w(TAG, "SMB $what: no session, connecting to ${c.host}")
            restore(c, null)
            return block(share ?: error("not connected"))
        }
        try {
            return block(sh)
        } catch (t: Throwable) {
            if (t is kotlinx.coroutines.CancellationException) throw t
            if (!isTransportFailure(t)) throw t
            val c = boundConn ?: throw t
            FileLog.w(TAG, "SMB $what failed (${t.javaClass.simpleName}: ${t.message}) — reconnecting to ${c.host}")
            restore(c, sh)
            return block(share ?: error("not connected"))
        }
    }

    /** Re-connect unless a concurrent caller already replaced [failed]. */
    private suspend fun restore(c: SmbConnection, failed: DiskShare?) = reconnectLock.withLock {
        val cur = share
        if (cur != null && cur !== failed) return@withLock // already restored
        connect(c)
    }

    private fun isTransportFailure(t: Throwable): Boolean = when (t) {
        // The server answered — the link is alive, retrying would repeat it —
        // unless it is telling us the session no longer exists.
        is SMBApiException -> t.statusCode == STATUS_USER_SESSION_DELETED
        else -> true // socket/transport errors, closed session, timeouts
    }

    fun disconnect() {
        synchronized(this) {
            try { share?.close() } catch (_: Exception) {}
            try { session?.close() } catch (_: Exception) {}
            try { client?.close() } catch (_: Exception) {}
            share = null; session = null; client = null
            connected = false; boundConnId = ""; boundConn = null
        }
    }

    private fun disconnectLocked() {
        try { share?.close() } catch (_: Exception) {}
        try { session?.close() } catch (_: Exception) {}
        try { client?.close() } catch (_: Exception) {}
        share = null; session = null; client = null
        connected = false; boundConnId = ""; boundConn = null
    }

    fun isBoundTo(id: String) = connected && boundConnId == id
}

/** Single open SMB file; thread-confined reads via synchronized. */
class SmbReadHandle internal constructor(
    private val file: com.hierynomus.smbj.share.File,
    val size: Long
) {
    private val lock = Any()
    fun readAt(offset: Long, buf: ByteArray, off: Int, len: Int): Int {
        synchronized(lock) {
            return file.read(buf, offset, off, len)
        }
    }
    fun close() { try { file.close() } catch (_: Exception) {} }
}

object SmbHolder {
    val manager = SmbManager()
}
