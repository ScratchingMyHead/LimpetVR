/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.RadioButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Modern 2D setup: Watch = top-level servers + This device, Connections =
 * servers+passwords (typed once, in 2D — never in VR), Display =
 * projection/stereo/optics/streaming prefs (mirrored in the VR settings page).
 */
class MainActivity : AppCompatActivity() {
    private lateinit var store: ConnectionStore
    private lateinit var settings: SettingsStore
    private var connections: MutableList<SmbConnection> = mutableListOf()

    private lateinit var container: ViewGroup
    private var watchView: View? = null
    private var connsView: View? = null
    private var displayView: View? = null

    // watch state: top level lists connections + This device
    private sealed interface WLoc {
        data object Root : WLoc
        data class Smb(val connId: String, val path: String) : WLoc
        data class Local(val dir: File) : WLoc
        data class Saf(val treeUri: String, val relPath: String, val label: String) : WLoc
    }
    private var wloc: WLoc = WLoc.Root
        // The 2D list is one face of the process-wide current directory:
        // every navigation publishes it, so the VR browser (and the next
        // showWatch) lands where the user actually is. SessionMemory only —
        // never persisted.
        set(v) {
            field = v
            SessionMemory.lastConnectionId = when (v) {
                is WLoc.Root -> ""
                is WLoc.Local -> "local:${v.dir.absolutePath}"
                is WLoc.Smb -> "smb:${v.connId}"
                is WLoc.Saf -> "saf:${v.treeUri}"
            }
            SessionMemory.lastPath = when (v) {
                is WLoc.Smb -> v.path
                is WLoc.Saf -> v.relPath
                else -> ""
            }
        }
    private var smbEntries: List<SmbEntry> = emptyList()
    private var localEntries: List<LocalFiles.LocalEntry> = emptyList()
    private lateinit var fileAdapter: FileAdapter

    private val permLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            val p = pendingSdPath; pendingSdPath = null
            if (p != null) openSdDirect(p) else openLocalRoot()
        } else Toast.makeText(this, "Videos permission needed for Internal Storage", Toast.LENGTH_LONG).show()
    }
    /** SD path awaiting the media permission, or the return from the
     *  All-files Settings toggle (auto-opens on the next render). */
    private var pendingSdPath: String? = null

    // SAF (SD card) folder picker state: volume awaiting a grant.
    private var pendingSafVolume: String? = null
    private val safPicker = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        if (uri != null) {
            SafFiles.takeGrant(this, uri)
            val cur = settings.safTrees.toMutableSet()
            cur += uri.toString()
            settings.safTrees = cur
            val label = SafFiles.removableVolumes(this).find { it.uuid == pendingSafVolume }?.desc ?: "SD card"
            pendingSafVolume = null
            wloc = WLoc.Saf(uri.toString(), "", label)
        } else pendingSafVolume = null
        renderWatch()
    }

    /** Open a file sent by another app (file manager → Open with → LimpetVR).
     *  The file's own folder becomes the current directory and its video
     *  siblings become the prev/next queue: without that a launch from a
     *  file manager leaves the directory unknown, so next/prev have nothing
     *  to step through and the 2D list can't open where the video lives. */
    private fun handleViewIntent(intent: Intent?) {
        val uri = intent?.data ?: return
        if (intent.action != Intent.ACTION_VIEW) return
        var name = "video"
        try {
            contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                null, null, null)?.use { c ->
                if (c.moveToFirst()) name = c.getString(0) ?: name
            }
        } catch (_: Exception) { }
        val f = viewIntentFile(uri)
        val parent = f?.parentFile
        if (f != null && parent != null && parent.isDirectory) {
            name = f.name
            wloc = WLoc.Local(parent)
            val qp = ArrayList(
                (parent.listFiles()?.filter { LocalFiles.isVideoFile(it) } ?: emptyList())
                    .sortedBy { it.name.lowercase() }.map { it.absolutePath }
            )
            var qi = qp.indexOf(f.absolutePath)
            if (qi < 0) { qp.add(0, f.absolutePath); qi = 0 }
            launchVr(uri.toString(), name, "local:${parent.absolutePath}", "", qp, qi)
        } else {
            // Opaque content:// with no filesystem path: play it, but leave
            // the remembered directory alone — we can't know this file's.
            launchVr(uri.toString(), name, "", "", arrayListOf(), -1)
        }
    }

    /** Best-effort filesystem path behind a VIEW uri: file://, the
     *  externalstorage documents provider (primary:DCIM/...), or a "_data"
     *  column on media-style content uris. Null when the provider won't say. */
    private fun viewIntentFile(uri: Uri): File? = try {
        when {
            uri.scheme == "file" || uri.scheme == null -> uri.path?.let { File(it) }
            uri.authority == "com.android.externalstorage.documents" -> {
                val doc = android.provider.DocumentsContract.getDocumentId(uri)
                val vol = doc.substringBefore(':', "")
                val rel = doc.substringAfter(':', "")
                val base = if (vol.equals("primary", true)) "/storage/emulated/0" else "/storage/$vol"
                File(if (rel.isEmpty()) base else "$base/$rel")
            }
            else -> {
                var path: String? = null
                contentResolver.query(uri, arrayOf("_data"), null, null, null)?.use { c ->
                    if (c.moveToFirst() && !c.isNull(0)) path = c.getString(0)
                }
                path?.let { File(it) }
            }
        }?.takeIf { it.exists() && it.isFile }
    } catch (_: Exception) { null }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleViewIntent(intent)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        FileLog.init(this)
        // Check media permission up front so local browsing never dead-ends.
        if (LocalFiles.needsPermission(this)) permLauncher.launch(LocalFiles.requestPermission())
        setContentView(R.layout.activity_main)
        store = ConnectionStore(this)
        settings = SettingsStore(this)
        connections = store.load()

        container = findViewById(R.id.container)
        val toolbar = findViewById<com.google.android.material.appbar.MaterialToolbar>(R.id.toolbar)
        toolbar.subtitle = "v${BuildConfig.VERSION_NAME}"
        toolbar.inflateMenu(R.menu.main_toolbar)
        toolbar.setOnMenuItemClickListener {
            when (it.itemId) {
                R.id.action_vr -> { enterVrBrowser(); true }
                R.id.action_web -> { enterVrWeb(); true }
                else -> false
            }
        }
        val nav = findViewById<BottomNavigationView>(R.id.bottomNav)
        nav.setOnItemSelectedListener {
            when (it.itemId) {
                R.id.tab_connections -> showConnections()
                R.id.tab_display -> showDisplay()
                R.id.tab_urls -> showUrls()
                else -> showWatch()
            }
            true
        }
        showWatch()
        handleViewIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        StreamProxy.start()
        connections = store.load()
        // Coming back from VR (or a file manager): the list must open on the
        // directory the video is playing from, not where it was left.
        wloc = wlocFromSession()
        // re-check the All-files toggle on return from Settings
        renderWatch()
    }

    // ---------- Watch ----------
    /** SessionMemory (the process-wide current directory) as a 2D location.
     *  Root when nothing is remembered or the remembered folder is gone. */
    private fun wlocFromSession(): WLoc {
        val ref = SessionMemory.lastConnectionId
        if (ref.isEmpty()) return WLoc.Root
        return when {
            ref.startsWith("local:") -> {
                val d = File(ref.removePrefix("local:"))
                if (d.isDirectory) WLoc.Local(d) else WLoc.Root
            }
            ref.startsWith("saf:") -> {
                val tree = ref.removePrefix("saf:")
                val label = (wloc as? WLoc.Saf)?.takeIf { it.treeUri == tree }?.label
                    ?: SafFiles.removableVolumes(this).find { it.uuid == tree }?.desc
                    ?: "SD card"
                WLoc.Saf(tree, SessionMemory.lastPath, label)
            }
            ref.startsWith("smb:") -> WLoc.Smb(ref.removePrefix("smb:"), SessionMemory.lastPath)
            // legacy bare connection id
            else -> WLoc.Smb(ref, SessionMemory.lastPath)
        }
    }

    private fun showWatch() {
        val v = watchView ?: LayoutInflater.from(this).inflate(R.layout.page_watch, container, false).also { watchView = it }
        swapTo(v)
        // Opening the file manager means showing the current directory —
        // wherever the last video was started (2D, VR or a file manager),
        // or wherever navigation last left it (the two are the same state).
        wloc = wlocFromSession()
        val list = v.findViewById<RecyclerView>(R.id.listFiles)
        list.layoutManager = LinearLayoutManager(this)
        fileAdapter = FileAdapter(
            onClick = { handleWatchClick(it) },
            onPlay = { playWatchEntry(it) },
            onScan = { scanWatchEntry(it) }
        )
        list.adapter = fileAdapter
        v.findViewById<View>(R.id.btnUp).setOnClickListener { watchHome() }
        renderWatch()
    }

    private sealed interface WatchEntry {
        data class RootConn(val conn: SmbConnection) : WatchEntry
        data object RootLocal : WatchEntry
        data class SdVolume(val desc: String, val uuid: String?) : WatchEntry
        data class Smb(val e: SmbEntry) : WatchEntry
        data class Local(val e: LocalFiles.LocalEntry) : WatchEntry
        data class Saf(val e: SafFiles.SafEntry, val treeUri: String, val relPath: String) : WatchEntry
        data object Up : WatchEntry
    }

    private fun openSafVolume(v: SafFiles.VolumeInfo) {
        val tree = SafFiles.grantedTree(this, v.uuid)
        if (tree != null) wloc = WLoc.Saf(tree, "", v.desc)
        else {
            pendingSafVolume = v.uuid
            try { safPicker.launch(null) } catch (_: Exception) {
                Toast.makeText(this, "No folder picker available", Toast.LENGTH_SHORT).show()
            }
            return
        }
        renderWatch()
    }

    private fun watchTitle(): String = when (val l = wloc) {
        is WLoc.Root -> "/"
        is WLoc.Saf -> "/${l.label}${if (l.relPath.isEmpty()) "" else "/${l.relPath}"}"
        is WLoc.Smb -> {
            val c = connections.find { it.id == l.connId }
            "/${c?.label ?: "?"}${if (l.path.isEmpty()) "" else "/${l.path}".replace('\\', '/')}"
        }
        is WLoc.Local -> {
            val (label, rel) = LocalFiles.rootTitle(this, l.dir)
            "/$label$rel"
        }
    }

    private fun renderWatch() {
        val v = watchView ?: return
        // Back from the All-files Settings toggle with a pending SD root:
        // drop straight into it (other file managers do the same) — unless
        // the user has navigated elsewhere meanwhile.
        val pend = pendingSdPath
        if (pend != null && wloc is WLoc.Root) {
            pendingSdPath = null
            val d = File(pend)
            if (!LocalFiles.needsPermission(this) && d.isDirectory && !LocalFiles.sdBlocked(this, d))
                wloc = WLoc.Local(d)
        }
        v.findViewById<TextView>(R.id.txtPath).text = watchTitle()
        when (val l = wloc) {
            is WLoc.Root -> {
                val items = mutableListOf<WatchEntry>()
                connections.forEach { items += WatchEntry.RootConn(it) }
                items += WatchEntry.RootLocal
                // SD via direct File (uuid field carries the absolute path);
                // SAF rows only survive where direct access is blocked or
                // a legacy grant already exists.
                for (vr in LocalFiles.volumeRoots(this).filter { !it.isPrimary }) {
                    val blocked = LocalFiles.sdBlocked(this, vr.dir)
                    items += WatchEntry.SdVolume(
                        if (blocked) "${vr.label} (tap to grant access)" else vr.label,
                        vr.dir.absolutePath)
                }
                for (v in SafFiles.removableVolumes(this)) {
                    if (!LocalFiles.needsFullAccess() && SafFiles.grantedTree(this, v.uuid) == null) continue
                    items += WatchEntry.SdVolume(v.desc, v.uuid)
                }
                fileAdapter.submit(items)
            }
            is WLoc.Saf -> {
                fileAdapter.submit(listOf(WatchEntry.Up))
                lifecycleScope.launch(Dispatchers.IO) {
                    val kids = try {
                        SafFiles.list(this@MainActivity, Uri.parse(l.treeUri), l.relPath)
                    } catch (t: Throwable) { emptyList() }
                    val items = mutableListOf<WatchEntry>(WatchEntry.Up)
                    kids.forEach { items += WatchEntry.Saf(it, l.treeUri, l.relPath) }
                    withContext(Dispatchers.Main) {
                        if (wloc == l) fileAdapter.submit(items)
                    }
                }
            }
            is WLoc.Smb -> {
                val conn = connections.find { it.id == l.connId }
                if (conn == null) { wloc = WLoc.Root; renderWatch(); return }
                fileAdapter.submit(listOf(WatchEntry.Up))
                lifecycleScope.launch(Dispatchers.IO) {
                    try {
                        if (!SmbHolder.manager.isBoundTo(conn.id)) SmbHolder.manager.connect(conn)
                        val list = SmbHolder.manager.list(l.path)
                        smbEntries = list
                        val items = mutableListOf<WatchEntry>(WatchEntry.Up)
                        list.forEach { items += WatchEntry.Smb(it) }
                        withContext(Dispatchers.Main) {
                            if (wloc == l) fileAdapter.submit(items)
                        }
                    } catch (t: Throwable) {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "SMB error: ${t.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            }
            is WLoc.Local -> {
                if (LocalFiles.needsPermission(this)) {
                    permLauncher.launch(LocalFiles.requestPermission())
                    wloc = WLoc.Root; renderWatch(); return
                }
                lifecycleScope.launch(Dispatchers.IO) {
                    val kids = try { LocalFiles.list(l.dir) } catch (t: Throwable) { emptyList() }
                    localEntries = kids
                    val items = mutableListOf<WatchEntry>(WatchEntry.Up)
                    kids.forEach { items += WatchEntry.Local(it) }
                    withContext(Dispatchers.Main) {
                        if (wloc == l) fileAdapter.submit(items)
                    }
                }
            }
        }
    }

    private fun watchUp() {
        wloc = when (val l = wloc) {
            is WLoc.Root -> WLoc.Root
            is WLoc.Smb -> if (l.path.isEmpty()) WLoc.Root else WLoc.Smb(l.connId, l.path.substringBeforeLast('\\', ""))
            is WLoc.Saf -> if (l.relPath.isEmpty()) WLoc.Root
                else WLoc.Saf(l.treeUri, l.relPath.substringBeforeLast('/', ""), l.label)
            is WLoc.Local -> {
                val isRoot = LocalFiles.volumeRoots(this).any { it.dir == l.dir }
                if (isRoot || l.dir.parentFile == null) WLoc.Root else WLoc.Local(l.dir.parentFile!!)
            }
        }
        renderWatch()
    }

    /** Top-level home button: jump straight to the server list. */
    private fun watchHome() {
        wloc = WLoc.Root
        renderWatch()
    }

    private fun handleWatchClick(e: WatchEntry) {
        when (e) {
            is WatchEntry.Up -> watchUp()
            is WatchEntry.RootConn -> { wloc = WLoc.Smb(e.conn.id, ""); renderWatch() }
            is WatchEntry.RootLocal -> openLocalRoot()
            // SdVolume.uuid carries an absolute path for direct-File SD
            // rows, or a volume uuid for legacy SAF rows.
            is WatchEntry.SdVolume -> {
                val u = e.uuid ?: ""
                if (u.startsWith("/")) openSdDirect(u) else openSafVolume(SafFiles.VolumeInfo(e.uuid, e.desc))
            }
            is WatchEntry.Smb -> if (e.e.isDir) { wloc = WLoc.Smb((wloc as WLoc.Smb).connId, e.e.path); renderWatch() }
            is WatchEntry.Local -> if (e.e.isDir) { wloc = WLoc.Local(e.e.file); renderWatch() }
            is WatchEntry.Saf -> if (e.e.isDir) {
                val child = if (e.relPath.isEmpty()) e.e.name else "${e.relPath}/${e.e.name}"
                wloc = WLoc.Saf(e.treeUri, child, (wloc as? WLoc.Saf)?.label ?: "SD card")
                renderWatch()
            }
        }
    }

    private fun openLocalRoot() {
        if (LocalFiles.needsPermission(this)) {
            permLauncher.launch(LocalFiles.requestPermission()); return
        }
        wloc = WLoc.Local(LocalFiles.externalRoot()); renderWatch()
    }

    /** Open an SD volume root directly; fires the media permission, then
     *  the All-files access dialog, only when each is actually missing. */
    private fun openSdDirect(path: String) {
        if (LocalFiles.needsPermission(this)) {
            pendingSdPath = path
            permLauncher.launch(LocalFiles.requestPermission()); return
        }
        val dir = java.io.File(path)
        if (LocalFiles.sdBlocked(this, dir)) {
            pendingSdPath = path
            showSdAccessDialog(path); return
        }
        pendingSdPath = null
        wloc = WLoc.Local(dir); renderWatch()
    }

    /** Why-tap SD flow: direct File browsing needs the All-files toggle
     *  (a Settings page — Android offers no popup for this grant), so say
     *  so up front and offer the SAF folder picker as the fallback. */
    private fun showSdAccessDialog(path: String) {
        val dir = java.io.File(path)
        val uuid = dir.name // XXXX-XXXX volume id; matches SAF tree docIds
        val label = SafFiles.removableVolumes(this).find { it.uuid == uuid }?.desc
            ?: LocalFiles.volumeRoots(this).find { it.dir.absolutePath == dir.absolutePath }?.label
            ?: "SD card ($uuid)"
        MaterialAlertDialogBuilder(this)
            .setTitle("SD card access")
            .setMessage("To browse the whole SD card directly, turn on " +
                "\"Allow all files access\" for LimpetVR on the next screen, then come back here.\n\n" +
                "Or pick a single folder instead — no full access needed.")
            .setPositiveButton("Open Settings") { _, _ -> LocalFiles.requestFullAccess(this) }
            .setNegativeButton("Pick a folder") { _, _ ->
                pendingSdPath = null
                openSafVolume(SafFiles.VolumeInfo(uuid, label))
            }
            .setNeutralButton("Cancel", null)
            .show()
    }

    private fun playWatchEntry(e: WatchEntry) {
        when (e) {
            is WatchEntry.SdVolume -> {
                val u = e.uuid ?: ""
                if (u.startsWith("/")) openSdDirect(u) else openSafVolume(SafFiles.VolumeInfo(e.uuid, e.desc))
            }
            is WatchEntry.Saf -> {
                if (!SafFiles.isVideo(e.e)) { Toast.makeText(this, "Only video files play", Toast.LENGTH_SHORT).show(); return }
                val sibs = try {
                    SafFiles.list(this, Uri.parse(e.treeUri), e.relPath).filter { SafFiles.isVideo(it) }
                } catch (t: Throwable) { emptyList() }
                val qp = ArrayList(sibs.map { it.uri.toString() })
                var qi = qp.indexOf(e.e.uri.toString())
                if (qi < 0) { qp.add(0, e.e.uri.toString()); qi = 0 }
                launchVr(e.e.uri.toString(), e.e.name, "saf:${e.treeUri}", e.relPath, qp, qi)
            }
            is WatchEntry.Smb -> {
                if (!e.e.isVideo()) { Toast.makeText(this, "Only video files play", Toast.LENGTH_SHORT).show(); return }
                val connId = (wloc as? WLoc.Smb)?.connId ?: return
                val curPath = (wloc as WLoc.Smb).path
                lifecycleScope.launch(Dispatchers.IO) {
                    try {
                        val probe = SmbHolder.manager.openRead(e.e.path)
                        val size = probe.size
                        probe.close()
                        val url = StreamProxy.register(
                            opener = { kotlinx.coroutines.runBlocking { SmbHolder.manager.openRead(e.e.path) } },
                            size = size,
                            displayName = e.e.name
                        )
                        withContext(Dispatchers.Main) {
                            val qp = ArrayList(smbEntries.filter { it.isVideo() }.map { it.path })
                            launchVr(url, e.e.name, connId, curPath, qp, qp.indexOf(e.e.path))
                        }
                    } catch (t: Throwable) {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "Open failed: ${t.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            }
            is WatchEntry.Local -> {
                if (e.e.isDir) { wloc = WLoc.Local(e.e.file); renderWatch(); return }
                val uri = Uri.fromFile(e.e.file).toString()
                val qp = ArrayList(localEntries.filter { !it.isDir }.map { it.file.absolutePath })
                launchVr(uri, e.e.name, "local:${e.e.file.parent}", "", qp, qp.indexOf(e.e.file.absolutePath))
            }
            else -> {}
        }
    }

    /** Sample-level file probe: per-track timestamps/gaps via proxy or file.
     *  Shows a dialog + logs to FileLog. Answers "is this file defective?"
     *  without decoding anything. */
    private fun scanWatchEntry(e: WatchEntry) {
        Toast.makeText(this, "Scanning samples…", Toast.LENGTH_SHORT).show()
        lifecycleScope.launch(Dispatchers.IO) {
            var proxyUrl: String? = null
            try {
                val uri: android.net.Uri
                val label: String
                when (e) {
                    is WatchEntry.Smb -> {
                        if (e.e.isDir) return@launch
                        val probe = SmbHolder.manager.openRead(e.e.path)
                        val size = probe.size
                        probe.close()
                        proxyUrl = StreamProxy.register(
                            opener = { kotlinx.coroutines.runBlocking { SmbHolder.manager.openRead(e.e.path) } },
                            size = size,
                            displayName = e.e.name
                        )
                        uri = android.net.Uri.parse(proxyUrl)
                        label = "smb:${e.e.path}"
                    }
                    is WatchEntry.Local -> {
                        if (e.e.isDir) return@launch
                        uri = Uri.fromFile(e.e.file)
                        label = e.e.file.absolutePath
                    }
                    else -> return@launch
                }
                val res = MediaScan.scan(this@MainActivity, uri, label)
                val text = MediaScan.summarize(res)
                FileLog.i("LimpetVR-scan", "\n$text")
                withContext(Dispatchers.Main) {
                    MaterialAlertDialogBuilder(this@MainActivity)
                        .setTitle("Sample scan")
                        .setMessage(text)
                        .setPositiveButton("OK", null)
                        .show()
                }
            } catch (t: Throwable) {
                FileLog.w("LimpetVR-scan", "scan failed: ${t.message}")
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Scan failed: ${t.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                proxyUrl?.let { StreamProxy.unregisterByUrl(it) }
            }
        }
    }

    private fun launchVr(url: String, name: String, connRef: String, path: String,
                         queuePaths: ArrayList<String> = arrayListOf(), queueIndex: Int = -1) {
        val i = Intent(this, VrPlayerActivity::class.java).apply {
            putExtra(VrPlayerActivity.EXTRA_URL, url)
            putExtra(VrPlayerActivity.EXTRA_NAME, name)
            putExtra(VrPlayerActivity.EXTRA_PROJ, settings.projection.name)
            putExtra(VrPlayerActivity.EXTRA_STEREO, settings.stereo.name)
            putExtra(VrPlayerActivity.EXTRA_CONN_ID, connRef)
            putExtra(VrPlayerActivity.EXTRA_PATH, path)
            putExtra(VrPlayerActivity.EXTRA_QUEUE_PATHS, queuePaths)
            putExtra(VrPlayerActivity.EXTRA_QUEUE_INDEX, queueIndex)
        }
        startActivity(i)
    }

    private fun enterVrBrowser() {
        val connRef = when (val l = wloc) {
            is WLoc.Smb -> "smb:${l.connId}"
            is WLoc.Local -> "local:${l.dir.absolutePath}"
            is WLoc.Saf -> "saf:${l.treeUri}"
            is WLoc.Root -> ""
        }
        val path = when (val l = wloc) {
            is WLoc.Smb -> l.path
            is WLoc.Saf -> l.relPath
            else -> ""
        }
        val i = Intent(this, VrPlayerActivity::class.java).apply {
            putExtra(VrPlayerActivity.EXTRA_PROJ, settings.projection.name)
            putExtra(VrPlayerActivity.EXTRA_STEREO, settings.stereo.name)
            putExtra(VrPlayerActivity.EXTRA_CONN_ID, connRef)
            putExtra(VrPlayerActivity.EXTRA_PATH, path)
        }
        startActivity(i)
    }

    /** "Enter Web": the cardboard display straight into the browser, with
     *  the same optics the 2D screen is set to. */
    private fun enterVrWeb() {
        startActivity(Intent(this, VrPlayerActivity::class.java).apply {
            putExtra(VrPlayerActivity.EXTRA_WEB, true)
            putExtra(VrPlayerActivity.EXTRA_PROJ, settings.projection.name)
            putExtra(VrPlayerActivity.EXTRA_STEREO, settings.stereo.name)
        })
    }

    // ---------- URLs (bookmarks for the VR web view) ----------
    private var urlsView: View? = null
    private var bookmarks: MutableList<WebBookmark> = mutableListOf()
    private val bookmarkStore by lazy { BookmarkStore(this) }

    private fun showUrls() {
        val v = urlsView ?: LayoutInflater.from(this).inflate(R.layout.page_urls, container, false)
            .also { urlsView = it }
        swapTo(v)
        v.findViewById<View>(R.id.btnAddUrl).setOnClickListener { editBookmark(null) }
        renderUrls(v)
    }

    private fun renderUrls(root: View) {
        bookmarks = bookmarkStore.load()
        val list = root.findViewById<android.widget.LinearLayout>(R.id.listUrls)
        list.removeAllViews()
        if (bookmarks.isEmpty()) {
            list.addView(android.widget.TextView(this).apply {
                text = "No bookmarks yet. Add one below — it will show up in the VR web panel."
                textSize = 14f; alpha = 0.7f; setPadding(4, 12, 4, 12)
            })
            return
        }
        val pad = (10 * resources.displayMetrics.density).toInt()
        for (b in bookmarks) {
            val row = android.widget.LinearLayout(this).apply {
                orientation = android.widget.LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                setPadding(4, pad, 4, pad)
                setBackgroundResource(android.R.drawable.list_selector_background)
                isClickable = true
                setOnClickListener { editBookmark(b) }
            }
            row.addView(android.widget.TextView(this).apply {
                text = b.title.ifBlank { b.url }
                textSize = 16f
                maxLines = 1
                layoutParams = android.widget.LinearLayout.LayoutParams(0,
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            })
            row.addView(android.widget.TextView(this).apply {
                text = "✕"
                textSize = 18f
                setPadding(pad * 2, 0, pad, 0)
                setOnClickListener {
                    bookmarks = bookmarks.filterNot { it.id == b.id }.toMutableList()
                    bookmarkStore.save(bookmarks)
                    renderUrls(root)
                }
            })
            list.addView(row)
            list.addView(android.widget.TextView(this).apply {
                text = b.url
                textSize = 12f; alpha = 0.6f; setPadding(6, 0, 6, 0)
            })
        }
    }

    private fun editBookmark(existing: WebBookmark?) {
        val dlg = LayoutInflater.from(this).inflate(R.layout.dialog_url, null)
        val name = dlg.findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.editUrlName)
        val addr = dlg.findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.editUrlAddr)
        if (existing != null) { name.setText(existing.title); addr.setText(existing.url) }
        MaterialAlertDialogBuilder(this)
            .setTitle(if (existing == null) "Add bookmark" else "Edit bookmark")
            .setView(dlg)
            .setPositiveButton("Save") { _, _ ->
                val url = normalizeUrl(addr.text.toString())
                if (url.isBlank()) {
                    Toast.makeText(this, "Enter a URL", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                val list = bookmarkStore.load()
                val id = existing?.id ?: java.util.UUID.randomUUID().toString()
                list.removeAll { it.id == id }
                list.add(WebBookmark(id, name.text.toString().trim(), url))
                bookmarkStore.save(list)
                renderUrls(urlsView ?: return@setPositiveButton)
            }
            .setNeutralButton(if (existing != null) "Delete" else null) { _, _ ->
                if (existing != null) {
                    bookmarkStore.save(bookmarkStore.load().filterNot { it.id == existing.id })
                    renderUrls(urlsView ?: return@setNeutralButton)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
            .also { it.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE) }
    }

    // ---------- Connections ----------
    private fun showConnections() {
        val v = connsView ?: LayoutInflater.from(this).inflate(R.layout.page_connections, container, false).also { connsView = it }
        swapTo(v)
        val list = v.findViewById<RecyclerView>(R.id.listConnections)
        list.layoutManager = LinearLayoutManager(this)
        val adapter = ConnAdapter(
            onUse = { c -> wloc = WLoc.Smb(c.id, ""); watchView?.let { showWatch() } ?: showWatch() },
            onDelete = { c ->
                connections.removeAll { it.id == c.id }
                store.save(connections)
                showConnections()
            }
        )
        list.adapter = adapter
        adapter.submit(connections.toList())
        v.findViewById<View>(R.id.btnAdd).setOnClickListener { editConnection(null) { showConnections(); watchView = null } }
    }

    private fun editConnection(existing: SmbConnection?, onSaved: () -> Unit) {
        val dlg = LayoutInflater.from(this).inflate(R.layout.dialog_connection, null)
        fun txt(id: Int) = dlg.findViewById<TextInputEditText>(id)
        if (existing != null) {
            txt(R.id.editName).setText(existing.name); txt(R.id.editHost).setText(existing.host)
            txt(R.id.editShare).setText(existing.share); txt(R.id.editDomain).setText(existing.domain)
            txt(R.id.editUser).setText(existing.username); txt(R.id.editPass).setText(existing.password)
        }
        MaterialAlertDialogBuilder(this)
            .setTitle(if (existing == null) "Add SMB connection" else "Edit connection")
            .setView(dlg)
            .setPositiveButton("Save") { _, _ ->
                val c = SmbConnection(
                    id = existing?.id ?: java.util.UUID.randomUUID().toString(),
                    name = txt(R.id.editName).text.toString().trim(),
                    host = txt(R.id.editHost).text.toString().trim(),
                    share = txt(R.id.editShare).text.toString().trim(),
                    domain = txt(R.id.editDomain).text.toString().trim(),
                    username = txt(R.id.editUser).text.toString().trim(),
                    password = txt(R.id.editPass).text.toString()
                )
                if (c.host.isBlank() || c.share.isBlank()) {
                    Toast.makeText(this, "Host and share are required", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                connections.removeAll { it.id == c.id }
                connections += c
                store.save(connections)
                onSaved()
            }
            .setNegativeButton("Cancel", null)
            .show()
            .also { it.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE) }
    }

    // ---------- Display ----------
    private fun showDisplay() {
        val v = displayView ?: LayoutInflater.from(this).inflate(R.layout.page_display, container, false).also { displayView = it }
        swapTo(v)
        val projIds = mapOf(
            Projection.FLAT to R.id.modeFlat, Projection.DEG180 to R.id.mode180,
            Projection.DEG220 to R.id.mode220, Projection.DEG270 to R.id.mode270,
            Projection.DEG360 to R.id.mode360, Projection.FISHEYE to R.id.modeFisheye
        )
        v.findViewById<RadioButton>(projIds[settings.projection]!!).isChecked = true
        val stereoIds = mapOf(Stereo.MONO to R.id.stereoMono, Stereo.SBS to R.id.stereoSbs, Stereo.TB to R.id.stereoTb)
        v.findViewById<RadioButton>(stereoIds[settings.stereo]!!).isChecked = true
        v.findViewById<android.widget.RadioGroup>(R.id.groupProjection).setOnCheckedChangeListener { _, id ->
            settings.projection = when (id) {
                R.id.modeFlat -> Projection.FLAT; R.id.mode220 -> Projection.DEG220
                R.id.mode270 -> Projection.DEG270; R.id.mode360 -> Projection.DEG360
                R.id.modeFisheye -> Projection.FISHEYE; else -> Projection.DEG180
            }
        }
        v.findViewById<android.widget.RadioGroup>(R.id.groupStereo).setOnCheckedChangeListener { _, id ->
            settings.stereo = when (id) {
                R.id.stereoSbs -> Stereo.SBS; R.id.stereoTb -> Stereo.TB; else -> Stereo.MONO
            }
        }
        bindSlider(v, R.id.sliderFov, R.id.lblFov, settings.fovScale, "FOV scale", "×") { settings.fovScale = it }
        bindSlider(v, R.id.sliderScreenSize, R.id.lblScreenSize, settings.screenSize, "Screen size (Flat only)", "×") { settings.screenSize = it }
        bindSlider(v, R.id.sliderCurve, R.id.lblScreenCurve, settings.screenCurve * 100f, "Screen curve (Flat only)", "%") { settings.screenCurve = it / 100f }
        bindSlider(v, R.id.sliderIpd, R.id.lblIpd, settings.ipdMm, "Eye separation (IPD)", "mm") { settings.ipdMm = it }
        bindSlider(v, R.id.sliderZoom, R.id.lblZoom, settings.videoZoom, "Video size (zoom)", "×") { settings.videoZoom = it }
        bindSlider(v, R.id.sliderSkip, R.id.lblSkip, settings.skipSecs.toFloat(), "Skip forward/back", "s") { settings.skipSecs = it.toInt() }
        bindSlider(v, R.id.sliderLensK1, R.id.lblLensK1, settings.lensK1, "Lens distortion k1", "") { settings.lensK1 = it }
        bindSlider(v, R.id.sliderLensK2, R.id.lblLensK2, settings.lensK2, "Lens distortion k2", "") { settings.lensK2 = it }
        bindSlider(v, R.id.sliderLensStrength, R.id.lblLensStrength, settings.lensStrength, "Lens strength", "×") { settings.lensStrength = it }
        bindSlider(v, R.id.sliderMenuAngleUp, R.id.lblMenuAngleUp, settings.menuAngleUp, "Play-menu look-up angle", "°") { settings.menuAngleUp = it }
        bindSlider(v, R.id.sliderMenuAngleDown, R.id.lblMenuAngleDown, -settings.menuAngleDown, "Play-menu look-down angle", "°") { settings.menuAngleDown = -it }
        bindSlider(v, R.id.sliderPanel, R.id.lblPanel, settings.panelDistM, "Browser panel distance", "m") { settings.panelDistM = it }
        val dwell = v.findViewById<Slider>(R.id.sliderDwell)
        dwell.value = snapToGrid(settings.dwellMs.toFloat(), dwell.valueFrom, dwell.valueTo, dwell.stepSize)
        v.findViewById<TextView>(R.id.lblDwell).text = "Gaze select delay — ${settings.dwellMs} ms"
        dwell.clearOnChangeListeners()
        dwell.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                settings.dwellMs = value.toLong()
                v.findViewById<TextView>(R.id.lblDwell).text = "Gaze select delay — ${value.toLong()} ms"
            }
        }
        val swSwap = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchSwapEyes)
        swSwap.isChecked = settings.swapEyes
        swSwap.setOnCheckedChangeListener { _, b -> settings.swapEyes = b }
        val swPin = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchPin)
        swPin.isChecked = settings.pinVideo
        swPin.setOnCheckedChangeListener { _, b -> settings.pinVideo = b }
        val swSweep = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchSweep)
        swSweep.isChecked = settings.testSweep
        swSweep.setOnCheckedChangeListener { _, b -> settings.testSweep = b }
        val swSw = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchSwDecode)
        swSw.isChecked = settings.softwareDecode
        swSw.setOnCheckedChangeListener { _, b ->
            settings.softwareDecode = b
            Toast.makeText(this, "Takes effect on next video", Toast.LENGTH_SHORT).show()
        }
        val swCirc = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchCircleRecenter)
        swCirc.isChecked = settings.circleRecenter
        swCirc.setOnCheckedChangeListener { _, b -> settings.circleRecenter = b }
        val swTip = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchTooltip)
        swTip.isChecked = settings.enableTooltip
        swTip.setOnCheckedChangeListener { _, b -> settings.enableTooltip = b }
        val swDist = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchDistortion)
        swDist.isChecked = !settings.disableDist
        swDist.setOnCheckedChangeListener { _, b -> settings.disableDist = !b }
        val swHq = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchPanoHq)
        swHq.isChecked = settings.panoQuality == "vertexhq"
        swHq.setOnCheckedChangeListener { _, b -> settings.panoQuality = if (b) "vertexhq" else "vertex" }
        val sw = v.findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchStartInVrBrowser)
        sw.isChecked = settings.startInVrBrowser
        sw.setOnCheckedChangeListener { _, b -> settings.startInVrBrowser = b }
        val buf = v.findViewById<TextInputEditText>(R.id.editBufferKb)
        if (buf.text.isNullOrEmpty()) buf.setText(settings.bufferKb.toString())
        buf.setOnFocusChangeListener { _, has -> if (!has) buf.text?.toString()?.toIntOrNull()?.let { settings.bufferKb = it.coerceIn(32, 2048) } }
    }

    private fun bindSlider(root: View, sliderId: Int, labelId: Int, cur: Float, name: String, unit: String, onSet: (Float) -> Unit) {
        val s = root.findViewById<Slider>(sliderId)
        val l = root.findViewById<TextView>(labelId)
        // Snap to the slider grid: values saved elsewhere (e.g. VR +/- steps
        // or float drift like 1.6000001) may sit off-grid, and Material Slider
        // throws IllegalStateException for those on set. Never crash setup.
        s.value = snapToGrid(cur, s.valueFrom, s.valueTo, s.stepSize)
        l.text = "$name — ${fmtNum(cur)} $unit"
        s.clearOnChangeListeners()
        s.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                onSet(value)
                l.text = "$name — ${fmtNum(value)} $unit"
            }
        }
    }

    private fun snapToGrid(v: Float, from: Float, to: Float, step: Float): Float {
        if (v.isNaN() || step <= 0f) return v.coerceIn(from, to)
        val n = kotlin.math.round((v - from) / step)
        return (from + n * step).coerceIn(from, to)
    }

    private fun fmtNum(f: Float): String =
        if (f >= 100 || f == f.toInt().toFloat()) f.toInt().toString() else String.format("%.1f", f)

    private fun swapTo(v: View) {
        container.removeAllViews()
        container.addView(v, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
    }

    // ---------- adapters ----------
    private class FileAdapter(
        val onClick: (WatchEntry) -> Unit,
        val onPlay: (WatchEntry) -> Unit,
        val onScan: (WatchEntry) -> Unit
    ) : RecyclerView.Adapter<FileAdapter.H>() {
        private var items: List<WatchEntry> = emptyList()
        fun submit(l: List<WatchEntry>) { items = l; notifyDataSetChanged() }
        class H(v: View) : RecyclerView.ViewHolder(v) {
            val name: TextView = v.findViewById(R.id.txtName)
            val meta: TextView = v.findViewById(R.id.txtMeta)
            val icon: TextView = v.findViewById(R.id.txtIcon)
            val scan: View = v.findViewById(R.id.btnScan)
        }
        override fun onCreateViewHolder(p: ViewGroup, t: Int) =
            H(LayoutInflater.from(p.context).inflate(R.layout.item_file, p, false))
        override fun getItemCount() = items.size
        override fun onBindViewHolder(h: H, pos: Int) {
            when (val e = items[pos]) {
                is WatchEntry.Up -> {
                    h.name.text = ".. (up)"; h.meta.text = ""; h.icon.text = "📁"
                    h.scan.visibility = View.GONE
                    h.itemView.setOnClickListener { onClick(e) }
                }
                is WatchEntry.RootConn -> {
                    h.name.text = e.conn.label; h.meta.text = e.conn.unc; h.icon.text = "🗄"
                    h.scan.visibility = View.GONE
                    h.itemView.setOnClickListener { onClick(e) }
                }
                is WatchEntry.RootLocal -> {
                    h.name.text = "Internal Storage"; h.meta.text = "phone storage"; h.icon.text = "📱"
                    h.scan.visibility = View.GONE
                    h.itemView.setOnClickListener { onClick(e) }
                }
                is WatchEntry.SdVolume -> {
                    h.name.text = e.desc
                    h.meta.text = if ((e.uuid ?: "").startsWith("/")) e.uuid else "SD card"
                    h.icon.text = "💾"
                    h.scan.visibility = View.GONE
                    h.itemView.setOnClickListener { onClick(e) }
                }
                is WatchEntry.Saf -> {
                    h.name.text = e.e.name
                    h.meta.text = if (e.e.isDir) "Folder" else MainActivity.humanSize(e.e.size)
                    h.icon.text = if (e.e.isDir) "📁" else if (SafFiles.isVideo(e.e)) "🎬" else "📄"
                    h.scan.visibility = View.GONE
                    h.itemView.setOnClickListener { if (e.e.isDir) onClick(e) else onPlay(e) }
                }
                is WatchEntry.Smb -> {
                    h.name.text = e.e.name
                    h.meta.text = if (e.e.isDir) "Folder" else MainActivity.humanSize(e.e.size)
                    h.icon.text = if (e.e.isDir) "📁" else if (e.e.isVideo()) "🎬" else "📄"
                    h.scan.visibility = if (e.e.isVideo()) View.VISIBLE else View.GONE
                    h.scan.setOnClickListener { onScan(e) }
                    h.itemView.setOnClickListener { if (e.e.isDir) onClick(e) else if (e.e.isVideo()) onPlay(e) }
                }
                is WatchEntry.Local -> {
                    h.name.text = e.e.name
                    h.meta.text = if (e.e.isDir) "Folder" else MainActivity.humanSize(e.e.size)
                    h.icon.text = if (e.e.isDir) "📁" else "🎬"
                    h.scan.visibility = if (!e.e.isDir) View.VISIBLE else View.GONE
                    h.scan.setOnClickListener { onScan(e) }
                    h.itemView.setOnClickListener { if (e.e.isDir) onClick(e) else onPlay(e) }
                }
            }
        }
    }

    private class ConnAdapter(
        val onUse: (SmbConnection) -> Unit,
        val onDelete: (SmbConnection) -> Unit
    ) : RecyclerView.Adapter<ConnAdapter.H>() {
        private var items: List<SmbConnection> = emptyList()
        fun submit(l: List<SmbConnection>) { items = l; notifyDataSetChanged() }
        class H(v: View) : RecyclerView.ViewHolder(v) {
            val name: TextView = v.findViewById(R.id.txtConnName)
            val detail: TextView = v.findViewById(R.id.txtConnDetail)
        }
        override fun onCreateViewHolder(p: ViewGroup, t: Int) =
            H(LayoutInflater.from(p.context).inflate(R.layout.item_connection, p, false))
        override fun getItemCount() = items.size
        override fun onBindViewHolder(h: H, pos: Int) {
            val c = items[pos]
            h.name.text = c.label
            h.detail.text = "${c.unc}  •  ${if (c.username.isBlank()) "guest" else c.username}"
            h.itemView.findViewById<View>(R.id.btnUse).setOnClickListener { onUse(c) }
            h.itemView.findViewById<View>(R.id.btnDelete).setOnClickListener { onDelete(c) }
        }
    }

    companion object {
        fun humanSize(n: Long): String {
            if (n < 1024) return "$n B"
            val kb = n / 1024.0
            if (kb < 1024) return String.format("%.0f KB", kb)
            val mb = kb / 1024.0
            if (mb < 1024) return String.format("%.1f MB", mb)
            return String.format("%.2f GB", mb / 1024.0)
        }
    }
}
