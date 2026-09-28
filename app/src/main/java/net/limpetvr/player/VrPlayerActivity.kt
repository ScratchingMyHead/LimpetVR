/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.graphics.drawable.ColorDrawable
import android.view.View
import android.view.ViewGroup
import android.widget.RelativeLayout
import com.google.vr.sdk.base.GvrView
import android.os.Bundle
import android.util.Log
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import kotlin.math.roundToInt
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.Tracks
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.analytics.AnalyticsListener
import androidx.media3.exoplayer.source.LoadEventInfo
import androidx.media3.exoplayer.source.MediaLoadData
import java.io.IOException
import org.json.JSONObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * VR player + in-headset browser.
 * Root level lists every SMB connection plus "This device".
 * A ⚙ Settings page exposes projection/stereo/FOV/IPD/zoom/dwell
 * without leaving VR. SMB files stream via the localhost Range proxy
 * (no download); local files play direct.
 */
class VrPlayerActivity : AppCompatActivity(), SensorEventListener {
    companion object {
        const val EXTRA_URL = "url"          // http proxy URL (SMB) or file:// URI (local)
        const val EXTRA_NAME = "name"
        const val EXTRA_PROJ = "proj"
        const val EXTRA_STEREO = "stereo"
        const val EXTRA_CONN_ID = "conn_id"  // smb:<id> | local:<abs path> | "" = root
        const val EXTRA_PATH = "path"
        const val EXTRA_QUEUE_PATHS = "queue_paths"  // sibling files for prev/next
        const val EXTRA_QUEUE_INDEX = "queue_index"
        private const val TAG = "LimpetVR"
        /** Tag on the top twin of the phone-alignment marker. */
        private const val ALIGN_TWIN_TAG = "align-marker-twin"
        /** Baseline fired already in this process (fresh-process detection).
         *  Rotation recreates the activity in-process and must keep live tuning. */
        private var baselineFiredProcess = false
        private const val BASELINE_EXPIRY_MS = 30L * 60L * 1000L
    }

    private lateinit var glView: GvrView
    private lateinit var renderer: VrRenderer
    private lateinit var txtStatus: TextView
    private var player: ExoPlayer? = null
    private lateinit var settings: SettingsStore
    private var connections: List<SmbConnection> = emptyList()

    // browser state
    private sealed interface Loc {
        data object Root : Loc
        data class Smb(val connId: String, val path: String) : Loc
        data class Local(val dir: File) : Loc
        data class Saf(val treeUri: String, val relPath: String, val label: String) : Loc
        data object SettingsPage : Loc
        data object ShapingPage : Loc
        data object Sensors : Loc
    }
    private var loc: Loc = Loc.Root
        // The VR browser is the other face of the process-wide current
        // directory: publishing every folder it opens means the 2D list
        // lands there on return (SessionMemory, never persisted).
        set(v) {
            field = v
            when (v) {
                is Loc.Root -> {
                    SessionMemory.lastConnectionId = ""
                    SessionMemory.lastPath = ""
                }
                is Loc.Local -> {
                    SessionMemory.lastConnectionId = "local:${v.dir.absolutePath}"
                    SessionMemory.lastPath = ""
                }
                is Loc.Smb -> {
                    SessionMemory.lastConnectionId = "smb:${v.connId}"
                    SessionMemory.lastPath = v.path
                }
                is Loc.Saf -> {
                    SessionMemory.lastConnectionId = "saf:${v.treeUri}"
                    SessionMemory.lastPath = v.relPath
                }
                // settings / shaping / sensors pages are not directories
                else -> {}
            }
        }
    // True when the settings page was opened from the in-video play menu:
    // backing out resumes the video instead of leaving to the servers.
    private var settingsFromVideo = false
    private var rows: List<Row> = emptyList()
    private data class Row(
        val label: String, val meta: String, val kind: Int,
        val smb: SmbEntry? = null, val local: File? = null,
        val action: String? = null, // for settings rows
        val slideKey: String? = null, // gaze slider id (settings rows)
        val slideMin: Float = 0f, val slideMax: Float = 1f, val slideVal: Float = 0f,
        val slideFmt: VrRenderer.SlideFormat? = null,
        val segLabels: List<String> = emptyList(), // segmented button row labels
        val segActions: List<String> = emptyList(), // one action per segment
        val segSelected: Int = -1, // currently active segment
        val previewMags: FloatArray? = null, // shaping preview: per-point magnitude 0..1
        val previewHull: IntArray? = null, // shaping preview: convex-hull point indices
        val previewN: Int = 9, // shaping preview grid size (n x n)
        val previewPos: FloatArray? = null, // shaping preview: absolute [x,y] per point, [0,1], y down
        val dead: Boolean = false, // rest zone: gaze may park here, nothing fires
        val saf: SafFiles.SafEntry? = null, // SAF (SD card) entries
        val safTree: String = "",
        val safRel: String = ""
    )

    /** Absolute preview positions from offsets: nominal + offset, y down. */
    private fun previewPos(ox: FloatArray, oy: FloatArray, n: Int): FloatArray {
        val pos = FloatArray(ox.size * 2)
        for (j in ox.indices) {
            pos[j * 2] = (j % n).toFloat() / (n - 1) + ox[j]
            pos[j * 2 + 1] = (j / n).toFloat() / (n - 1) + oy[j]
        }
        return pos
    }

    /** Authored warp shape from shaping/shapes.json: 9x9 absolute (u,v) grid,
     *  row-major, row 0 = top, v top->bottom. Converted to offsets at load
     *  (offset = point - nominal) in half-frame normalized units. */
    private data class UiShape(
        val id: String,
        val label: String,
        val n: Int,
        val ox: FloatArray, // x offsets, +right
        val oy: FloatArray, // y offsets, +down (grid space)
        val mags: FloatArray, // per-point magnitude normalized 0..1 by file max
        val maxMag: Float,
        val hull: IntArray // convex-hull indices over moved points
    )
    private var shapingShapes: List<UiShape> = emptyList()
    private var lastShapingKey = "\u0000"

    private fun slugShapeId(name: String): String {
        val s = name.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-')
        return s.ifEmpty { "shape" }
    }

    /** Load shaping grids from APK assets (copied from misc/shapes.json at build).
     *  Bad files/points are skipped with a log; never throws. */
    private fun loadShapingShapes() {
        try {
            val raw = assets.open("shaping/shapes.json").bufferedReader().use { it.readText() }
            val arr = JSONObject(raw).getJSONArray("shapes")
            val out = mutableListOf<UiShape>()
            val used = mutableSetOf<String>()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val name = o.optString("name", "shape ${i + 1}")
                var id = slugShapeId(name)
                var k = 2
                while (id in used) { id = "${slugShapeId(name)}-$k"; k++ }
                used += id
                val g = o.getJSONArray("grid")
                val count = g.length()
                val n = kotlin.math.sqrt(count.toDouble()).toInt()
                if (n < 2 || n * n != count) { Log.w(TAG, "shaping: $name bad grid size $count"); continue }
                val ox = FloatArray(count); val oy = FloatArray(count)
                var ok = true
                for (j in 0 until count) {
                    val pt = g.getJSONArray(j)
                    val x = pt.optDouble(0, Double.NaN); val y = pt.optDouble(1, Double.NaN)
                    if (x.isNaN() || y.isNaN()) { ok = false; break }
                    ox[j] = (x - (j % n).toDouble() / (n - 1)).toFloat().coerceIn(-1f, 1f)
                    oy[j] = (y - (j / n).toDouble() / (n - 1)).toFloat().coerceIn(-1f, 1f)
                }
                if (!ok) { Log.w(TAG, "shaping: $name bad point"); continue }
                var maxMag = 0f
                val mags = FloatArray(count)
                for (j in 0 until count) {
                    val m = kotlin.math.hypot(ox[j], oy[j])
                    mags[j] = m; if (m > maxMag) maxMag = m
                }
                val norm = if (maxMag > 1e-9f) FloatArray(count) { mags[it] / maxMag } else FloatArray(count)
                out += UiShape(id, name, n, ox, oy, norm, maxMag, convexHull(ox, oy, mags, n))
            }
            shapingShapes = out
            Log.i(TAG, "shaping: loaded ${out.size} shapes")
        } catch (t: Throwable) {
            Log.e(TAG, "shaping load failed", t)
            shapingShapes = emptyList()
        }
    }

    /** Minimum convex polygon (Andrew monotone chain) over moved points,
     *  in nominal grid coords + offset (display positions). Returns grid indices. */
    private fun convexHull(ox: FloatArray, oy: FloatArray, mags: FloatArray, n: Int): IntArray {
        data class P(val x: Double, val y: Double, val idx: Int)
        val pts = mutableListOf<P>()
        for (j in ox.indices) {
            if (mags[j] <= 1e-6f) continue
            pts += P((j % n).toDouble() / (n - 1) + ox[j], (j / n).toDouble() / (n - 1) + oy[j], j)
        }
        if (pts.size < 3) return pts.map { it.idx }.toIntArray()
        val s = pts.sortedWith(compareBy({ it.x }, { it.y }))
        fun cross(o: P, a: P, b: P) = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        val lower = mutableListOf<P>()
        for (p in s) { while (lower.size >= 2 && cross(lower[lower.size - 2], lower.last(), p) <= 0) lower.removeLast(); lower += p }
        val upper = mutableListOf<P>()
        for (p in s.reversed()) { while (upper.size >= 2 && cross(upper[upper.size - 2], upper.last(), p) <= 0) upper.removeLast(); upper += p }
        lower.removeLast(); upper.removeLast()
        return (lower + upper).map { it.idx }.toIntArray()
    }

    /** Normalized weighted average of enabled shapes (null = identity).
     *  Averages offsets in grid space; weights are 0..1. */
    private fun averagedShapeOffsets(): Triple<FloatArray, FloatArray, Int>? {
        val active = shapingShapes.filter { settings.shapeEnabled(it.id) }
        if (active.isEmpty()) return null
        val n = active[0].n
        val same = active.filter { it.n == n }
        var wsum = 0f
        for (s in same) wsum += settings.shapeWeight(s.id) / 100f
        if (wsum <= 0f) return null
        val ex = FloatArray(n * n); val ey = FloatArray(n * n)
        for (s in same) {
            // absolute strength (mean), matching bakeShaping: a lone shape
            // at 50% previews at half displacement, not full
            val w = settings.shapeWeight(s.id) / 100f / same.size
            for (j in ex.indices) { ex[j] += w * s.ox[j]; ey[j] += w * s.oy[j] }
        }
        return Triple(ex, ey, n)
    }

    private var connId: String = ""
    private var playUrl: String? = null
    private var playIsProxy = false
    // SAF (SD card) folder picker state: volume awaiting a grant.
    private var pendingSafVolume: String? = null
    private val safPicker = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        if (uri != null) {
            SafFiles.takeGrant(this, uri)
            val cur = settings.safTrees.toMutableSet()
            cur += uri.toString()
            settings.safTrees = cur
            val label = SafFiles.removableVolumes(this).find { it.uuid == pendingSafVolume }?.desc ?: "SD card"
            pendingSafVolume = null
            loc = Loc.Saf(uri.toString(), "", label)
        } else pendingSafVolume = null
        refresh()
    }

    private lateinit var sensors: SensorManager
    private var rotSensor: Sensor? = null
    private var useGameRv = false
    private var cmpSensor: Sensor? = null // A/B: second source for comparison
    private var oriLogCountdown = 0
    // Raw environment sensors for the debug overlay (snapshots, sensor thread).
    private var gyroSensor: Sensor? = null
    private var accelSensor: Sensor? = null
    private var magSensor: Sensor? = null
    @Volatile private var lastGyro = FloatArray(3)
    @Volatile private var lastAccel = FloatArray(3)
    @Volatile private var lastMag = FloatArray(3)
    @Volatile private var lastTrackM: FloatArray? = null
    private val sensorTick = android.os.Handler(android.os.Looper.getMainLooper())
    private var sensorTicking = false
    // Yaw-direction baseline captured on first debug-page refresh.
    private var yawBaseG: Int? = null
    private var yawBaseR: Int? = null
    private var yawBaseC: Int? = null
    // TURNDET baselines (page-open): gyro-integrated yaw + rendered yaw.
    private var turnBaseG: Double? = null
    private var turnBaseR: Double? = null
    // Flight recorder: raw sensor matrix + rendered forward per event.
    // File: getFilesDir()/trace-<ts>.csv. Open VR, turn head left-right once,
    // nod once, exit — then pull the file for analysis.
    private var traceQueue: java.util.concurrent.LinkedBlockingQueue<String>? = null
    private var traceThread: Thread? = null
    private var traceCount = 0
    private var traceFullLogged = false
    private val TRACE_MAX_LINES = 12000

    private fun startTrace() {
        try {
            val f = java.io.File(filesDir, "trace-${System.currentTimeMillis()}.csv")
            val q: java.util.concurrent.LinkedBlockingQueue<String> =
                java.util.concurrent.LinkedBlockingQueue()
            traceQueue = q
            traceCount = 0
            traceFullLogged = false
            q.put("t_ns," + (0..15).joinToString(",") { "r$it" } + ",fwdx,fwdy,fwdz," + (0..15).joinToString(",") { "c$it" } + ",gx,gy,gz,ax,ay,az\n")
            traceThread = kotlin.concurrent.thread(name = "limpetvr-trace", isDaemon = true) {
                try {
                    f.bufferedWriter().use { w ->
                        while (true) {
                            val line = q.take()
                            if (line == "EOF") break
                            w.write(line)
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "trace: ${e.message}")
                }
            }
            Log.i(TAG, "trace recording to ${f.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "trace start failed: ${e.message}")
        }
    }

    private fun traceEvent(tNs: Long, raw: FloatArray) {
        val q = traceQueue ?: return
        if (traceCount >= TRACE_MAX_LINES) {
            if (!traceFullLogged) {
                traceFullLogged = true
                Log.w(TAG, "trace full ($TRACE_MAX_LINES lines), stopping")
            }
            return
        }
        traceCount++
        val fwd = renderer.lastEffFwd
        val cmp = cmpListener.lastRaw
        val g = lastGyro
        val ac = lastAccel
        val sb = StringBuilder(480)
        sb.append(tNs)
        for (v in raw) sb.append(',').append(v)
        sb.append(',').append(fwd[0]).append(',').append(fwd[1]).append(',').append(fwd[2])
        if (cmp != null) { for (v in cmp) sb.append(',').append(v) }
        else { for (i in 0..15) sb.append(",NaN") }
        sb.append(',').append(g[0]).append(',').append(g[1]).append(',').append(g[2])
        sb.append(',').append(ac[0]).append(',').append(ac[1]).append(',').append(ac[2])
        sb.append('\n')
        q.offer(sb.toString())
    }

    private fun stopTrace() {
        traceQueue?.offer("EOF")
        traceQueue = null
        traceThread = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        FileLog.init(this)
        FileLog.i(TAG, "LimpetVR ${BuildConfig.VERSION_NAME} VrPlayerActivity start")
        val volProof = runCatching {
            val f = VrRenderer::class.java.getDeclaredField("frameAvailable")
            java.lang.reflect.Modifier.isVolatile(f.modifiers)
        }.getOrDefault(false)
        FileLog.i(TAG, "volatileProof frameAvailable=$volProof")
        android.util.Log.i(TAG, "volatileProof frameAvailable=$volProof")
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        hideSystemBars()
        setContentView(R.layout.activity_vr)
        settings = SettingsStore(this)
        // Startup baseline (§10): fresh install, fresh process, or 30-min
        // expiry. Rotation recreates in-process and keeps live tuning.
        if (!baselineFiredProcess) {
            baselineFiredProcess = true
            settings.fireBaseline(System.currentTimeMillis())
            FileLog.i(TAG, "startup baseline fired (fresh process)")
            Log.i(TAG, "startup baseline fired (fresh process)")
        }
        connections = ConnectionStore(this).load()
        loadShapingShapes()

        glView = findViewById(R.id.glView)
        txtStatus = findViewById(R.id.txtStatus)
        glView.setEGLContextClientVersion(2)
        renderer = VrRenderer(
            onBrowserActivate = { idx, frac -> runOnUiThread { activateRow(idx, frac) } },
            onMenuEvent = { e -> runOnUiThread { handleMenuEvent(e) } }
        )
        applyOptics()
        renderer.projection = runCatching { Projection.valueOf(intent.getStringExtra(EXTRA_PROJ) ?: settings.projection.name) }.getOrDefault(settings.projection)
        renderer.stereo = runCatching { Stereo.valueOf(intent.getStringExtra(EXTRA_STEREO) ?: settings.stereo.name) }.getOrDefault(settings.stereo)
        syncGvr() // projection decides the neck model; IPD/lens too
        glView.setRenderer(renderer)
        // StereoRenderer drives both eyes; the SDK handles lens warp.
        glView.setStereoModeEnabled(true)
        renderer.gvrView = glView
        tuneAlignmentMarker() // halve GVR's bottom marker, mirror one at the top
        glView.addOnLayoutChangeListener { v, _, _, _, _, _, _, _, _ ->
            renderer.lastWidth = v.width.coerceAtLeast(1)
            renderer.lastHeight = v.height.coerceAtLeast(1)
        }
        glView.setOnTouchListener { _, e ->
            // Tap recenters silently: the world snapping is its own feedback.
            // While the blue aim is armed the tap cancels it instead.
            if (e.action == MotionEvent.ACTION_UP && !renderer.cancelAim()) renderer.recenter("tap")
            true
        }

        sensors = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        // Prefer GAME rotation vector: gyro+accel only, no compass, so viewer
        // magnets / indoor metal can't corrupt yaw (GVR-style fusion behaves
        // the same way). Yaw has no absolute reference and drifts slowly — tap to
        // recenter. Fall back to the full rotation vector if absent.
        rotSensor = sensors.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensors.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        useGameRv = rotSensor?.type == Sensor.TYPE_GAME_ROTATION_VECTOR
        // A/B comparison: always listen to the OTHER source too (diagnostics only)
        cmpSensor = if (useGameRv) sensors.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            else sensors.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        gyroSensor = sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        accelSensor = sensors.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        magSensor = sensors.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        findViewById<android.view.View>(R.id.btnClose).setOnClickListener { finish() }

        connId = intent.getStringExtra(EXTRA_CONN_ID) ?: SessionMemory.lastConnectionId
        val url = intent.getStringExtra(EXTRA_URL)
        if (url != null) {
            playIsProxy = url.startsWith("http://127.0.0.1")
            // Direct launch (2D Watch play button): rebuild the prev/next
            // queue + remembered folder from the extras, exactly like the
            // in-VR play path does from its listing.
            val qPaths = intent.getStringArrayListExtra(EXTRA_QUEUE_PATHS) ?: arrayListOf()
            val qIndex = intent.getIntExtra(EXTRA_QUEUE_INDEX, -1)
            if (connId.startsWith("local:")) {
                val dir = connId.removePrefix("local:")
                SessionMemory.lastConnectionId = connId
                SessionMemory.lastPath = ""
                playQueue = qPaths.map { PlayItem.Local(File(it)) }
                playIndex = qIndex
                if (playIndex !in playQueue.indices) {
                    playIndex = playQueue.indexOfFirst {
                        (it as PlayItem.Local).f.absolutePath == File(dir, intent.getStringExtra(EXTRA_NAME) ?: "").absolutePath
                    }
                }
            } else if (connId.startsWith("saf:")) {
                val treeUri = connId.removePrefix("saf:")
                val relPath = intent.getStringExtra(EXTRA_PATH) ?: ""
                SessionMemory.lastConnectionId = connId
                SessionMemory.lastPath = relPath
                playQueue = qPaths.map {
                    PlayItem.Saf(it, android.net.Uri.parse(it).lastPathSegment ?: "video", treeUri, relPath)
                }
                playIndex = qIndex
                if (playIndex !in playQueue.indices) {
                    playIndex = playQueue.indexOfFirst { (it as? PlayItem.Saf)?.uri == url }
                }
            } else if (connId.isNotEmpty()) {
                val cleanId = connId.removePrefix("smb:")
                val dirPath = intent.getStringExtra(EXTRA_PATH) ?: SessionMemory.lastPath
                SessionMemory.lastConnectionId = "smb:$cleanId"
                SessionMemory.lastPath = dirPath
                playQueue = qPaths.map {
                    PlayItem.Smb(SmbEntry(it.substringAfterLast('\\'), it, false, -1), cleanId)
                }
                playIndex = qIndex
            }
            Log.i(TAG, "launch dir=$connId queue=${playQueue.size} idx=$playIndex")
            startPlayback(url, intent.getStringExtra(EXTRA_NAME) ?: "video")
        } else {
            loc = when {
                connId.startsWith("local:") -> Loc.Local(File(connId.removePrefix("local:")))
                connId.startsWith("smb:") -> Loc.Smb(connId.removePrefix("smb:"), intent.getStringExtra(EXTRA_PATH) ?: SessionMemory.lastPath)
                connId.isNotEmpty() -> Loc.Smb(connId, intent.getStringExtra(EXTRA_PATH) ?: SessionMemory.lastPath)
                else -> Loc.Root
            }
            enterBrowser()
        }
    }

    private fun applyOptics() {
        renderer.swapEyes = settings.swapEyes
        renderer.zoom = settings.videoZoom
        renderer.panelDistM = settings.panelDistM
        renderer.dwellMs = settings.dwellMs
        renderer.pinVideo = settings.pinVideo
        renderer.skipSecs = settings.skipSecs
        // Optics: FOV scale + screen size feed the per-eye projection and
        // the flat model matrix; IPD/lens coefficients go to the SDK.
        renderer.fovScale = settings.fovScale
        renderer.screenSize = settings.screenSize
        renderer.screenCurve = settings.screenCurve
        renderer.panoQuality = settings.panoQuality
        renderer.disableDist = settings.disableDist
        renderer.enableTooltip = settings.enableTooltip
        renderer.ipdM = settings.ipdMm / 1000f
        renderer.convTrimNdc = settings.convTrim

        // Shaping grids: feed active set only when content changed (mesh rebuild is keyed).
        val skey = shapingShapes.filter { settings.shapeEnabled(it.id) }
            .joinToString(";") { "${it.id}:${settings.shapeWeight(it.id)}" }
        if (skey != lastShapingKey) {
            lastShapingKey = skey
            renderer.shapingActive = shapingShapes.filter { settings.shapeEnabled(it.id) }.map {
                VrRenderer.ActiveShape(it.ox, it.oy, settings.shapeWeight(it.id) / 100f, it.n)
            }
            renderer.shapingRevision++
        }
        renderer.testSweep = settings.testSweep
        renderer.lensK1 = settings.lensK1
        renderer.lensK2 = settings.lensK2
        renderer.lensStrength = settings.lensStrength
        renderer.fisheyeRadiusScale = settings.fisheyeRadius
        renderer.fisheyeMirrorR = settings.fisheyeMirrorR
        renderer.menuAngleUp = settings.menuAngleUp
        renderer.menuAngleDown = settings.menuAngleDown
        renderer.menuSideUp = settings.menuTop
        renderer.circleGestureEnabled = settings.circleRecenter
        // Fisheye circle calibration needs the decoded frame's aspect.
        try {
            player?.videoFormat?.let { vf ->
                if (vf.width > 0 && vf.height > 0)
                    renderer.videoAspect = vf.width.toFloat() / vf.height.toFloat()
            }
        } catch (_: Exception) {}
        syncGvr()
    }

    /** GVR-side optics (§6/§10): viewer IPD, lens distortion coefficients
     *  (k1/k2 × strength), the distortion-correction toggle and the neck
     *  model — flat holds the screen fixed in view space, panoramas stay
     *  tracked. Runs from applyOptics so the 2-D settings page is live. */
    private fun syncGvr() {
        val g = renderer.gvrView ?: return
        runCatching {
            g.setNeckModelEnabled(renderer.projection == Projection.FLAT)
            g.setDistortionCorrectionEnabled(!settings.disableDist)
            val vp = g.gvrViewerParams
            vp.interLensDistance = renderer.ipdM
            vp.distortion.setCoefficients(floatArrayOf(
                settings.lensK1 * settings.lensStrength,
                settings.lensK2 * settings.lensStrength
            ))
            g.updateGvrViewerParams(vp)
            FileLog.i(TAG, "syncGvr ipd=${renderer.ipdM} dist=${!settings.disableDist} " +
                "k=(${settings.lensK1 * settings.lensStrength},${settings.lensK2 * settings.lensStrength}) " +
                "neck=${renderer.projection == Projection.FLAT}")
        }.onFailure { FileLog.i(TAG, "syncGvr failed: $it") }
    }

    /** GVR's phone-alignment marker is a 24 mm line pinned to the bottom
     *  centre of the screen (cardboard ui_layer.xml: id ui_alignment_marker,
     *  #b4b4b4, 2 dp thick). Request: half that length, plus an equal twin
     *  pinned to the top so both edges mark the phone centre. Runs after
     *  layout; the original length is captured once so repeat calls (and a
     *  UiLayer re-inflate) stay idempotent, and the twin carries a tag. */
    private var alignMarkerFullH = 0
    private fun tuneAlignmentMarker() {
        glView.post {
            val id = com.google.vr.cardboard.R.id.ui_alignment_marker
            val marker = glView.findViewById<View>(id) ?: window.decorView.findViewById<View>(id)
            if (marker == null) { Log.w(TAG, "alignment marker not found"); return@post }
            val lp = marker.layoutParams as? RelativeLayout.LayoutParams ?: return@post
            if (lp.height <= 0 || lp.height < lp.width) return@post // portrait variant: a bar
            if (alignMarkerFullH == 0) alignMarkerFullH = lp.height
            lp.height = alignMarkerFullH / 2
            marker.layoutParams = lp
            val parent = marker.parent as? ViewGroup ?: return@post
            if (parent.findViewWithTag<View>(ALIGN_TWIN_TAG) != null) return@post
            val color = (marker.background as? ColorDrawable)?.color ?: 0xFFB4B4B4.toInt()
            parent.addView(View(glView.context).apply {
                tag = ALIGN_TWIN_TAG
                setBackgroundColor(color)
                layoutParams = RelativeLayout.LayoutParams(lp.width, alignMarkerFullH / 2).apply {
                    addRule(RelativeLayout.CENTER_HORIZONTAL)
                    addRule(RelativeLayout.ALIGN_PARENT_TOP)
                }
            })
            FileLog.i(TAG, "alignment marker ${alignMarkerFullH}px -> ${alignMarkerFullH / 2}px + top twin")
        }
    }

    override fun onResume() {
        super.onResume()
        hideSystemBars()
        // 30-minute baseline expiry (§10). Re-entry inside a session keeps
        // live tuning (onCreate already handled the fresh-process fire).
        if (System.currentTimeMillis() - settings.baselineAt > BASELINE_EXPIRY_MS) {
            settings.fireBaseline(System.currentTimeMillis())
            FileLog.i(TAG, "startup baseline fired (30-min expiry)")
            Log.i(TAG, "startup baseline fired (30-min expiry)")
        }
        applyOptics()
        renderer.resetBasis("entry") // next sensor reading centers the view
        connections = ConnectionStore(this).load()
        // re-check the All-files toggle on return from Settings
        try { refresh() } catch (_: Exception) {}
        glView.onResume()
        tuneAlignmentMarker() // idempotent: covers a UiLayer (re)inflate
        rotSensor?.let {
            sensors.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            Log.i(TAG, "tracking via ${if (useGameRv) "GAME_ROTATION_VECTOR (no compass)" else "ROTATION_VECTOR"}")
        }
        cmpSensor?.let {
            sensors.registerListener(cmpListener, it, SensorManager.SENSOR_DELAY_GAME)
            Log.i(TAG, "A/B comparison via type=${it.type}")
        }
        gyroSensor?.let { sensors.registerListener(envListener, it, SensorManager.SENSOR_DELAY_GAME) }
        accelSensor?.let { sensors.registerListener(envListener, it, SensorManager.SENSOR_DELAY_GAME) }
        magSensor?.let { sensors.registerListener(envListener, it, SensorManager.SENSOR_DELAY_GAME) }
        startTrace()
        startSensorTick()
        startWatchTick()
    }

    /** 2Hz live refresh + 1Hz logcat mirror while the sensor page is open. */
    private val sensorTickTask = object : Runnable {
        var n = 0
        override fun run() {
            if (!sensorTicking) return
            if (loc == Loc.Sensors && renderer.mode == VrRenderer.Mode.BROWSER) {
                updatePeaks()
                showSensors()
                if (n++ % 2 == 0) logSensors()
            }
            sensorTick.postDelayed(this, 500)
        }
    }

    // Peak-hold telemetry: latch max gyro per axis + max gaze deflection from
    // page-open baseline, so motion can be read AFTER stopping (gyro shows
    // rate = zero when still; Euler rows are gimbal-unreliable in-viewer).
    private var peakGyro = FloatArray(3)
    private var peakBaseFwd: FloatArray? = null
    private var peakGazeDeg = 0.0
    private var peakGazeDir = ""

    private fun resetPeaks() {
        peakGyro = FloatArray(3)
        peakBaseFwd = null
        peakGazeDeg = 0.0
        peakGazeDir = ""
    }

    private fun updatePeaks() {
        val g = lastGyro
        for (i in 0..2) {
            val m = kotlin.math.abs(g[i])
            if (m > peakGyro[i]) peakGyro[i] = m
        }
        val e = renderer.effCopy()
        val f = floatArrayOf(-e[2], -e[6], -e[10])
        val base = peakBaseFwd
        if (base == null) {
            peakBaseFwd = f
            return
        }
        val dot = ((f[0]*base[0] + f[1]*base[1] + f[2]*base[2]) /
            (kotlin.math.sqrt((f[0]*f[0] + f[1]*f[1] + f[2]*f[2]).toDouble()) *
             kotlin.math.sqrt((base[0]*base[0] + base[1]*base[1] + base[2]*base[2]).toDouble()) + 1e-9))
            .coerceIn(-1.0, 1.0)
        val ang = Math.toDegrees(kotlin.math.acos(dot))
        if (ang > peakGazeDeg) {
            peakGazeDeg = ang
            // direction of deflection in screen terms (from baseline forward)
            val yawPart = Math.toDegrees(kotlin.math.atan2((f[0] - base[0]).toDouble(), (-(f[2] - base[2])).toDouble()))
            val pitPart = Math.toDegrees(kotlin.math.asin((f[1] - base[1]).toDouble().coerceIn(-1.0, 1.0)))
            peakGazeDir = (if (yawPart >= 0) "R" else "L") + (if (pitPart >= 0) "+up" else "+dn")
        }
    }

    private fun startSensorTick() {
        if (sensorTicking) return
        sensorTicking = true
        sensorTick.post(sensorTickTask)
    }

    private fun stopSensorTick() {
        sensorTicking = false
        sensorTick.removeCallbacks(sensorTickTask)
        sensorTick.removeCallbacks(watchTickTask)
    }

    // Playback stall watchdog: 2s tick while video plays. Distinguishes
    // BUFFERING-starved (state=BUFFERING, buffered~0 → link too slow) from
    // STUCK-ready (state=READY but position frozen with buffer → decoder/
    // render stall) from clean playback. The money log for freezes.
    // Also re-attaches the video surface if EGL recreated it under us
    // (orphaned surface = frozen picture, advancing position, no errors).
    private var attachedSurfaceGen = -1
    private var lastWatchFrames = 0L
    private var lastWatchConsumed = 0L
    private var lastWatchArrived = 0L
    @Volatile private var lastFrameBatchMs = 0L
    private val watchTickTask = object : Runnable {
        override fun run() {
            val p = player
            if (p != null && renderer.mode == VrRenderer.Mode.VIDEO) {
                if (attachedSurfaceGen != renderer.surfaceGen) {
                    attachedSurfaceGen = renderer.surfaceGen
                    val s = renderer.surface
                    if (s != null) {
                        p.setVideoSurface(s)
                        FileLog.i(TAG, "video surface (re-)attached gen=${renderer.surfaceGen}")
                    }
                }
                val state = when (p.playbackState) {
                    Player.STATE_IDLE -> "IDLE"
                    Player.STATE_BUFFERING -> "BUFFERING"
                    Player.STATE_READY -> "READY"
                    Player.STATE_ENDED -> "ENDED"
                    else -> "?${p.playbackState}"
                }
                val pos = p.currentPosition
                renderer.menuPosMs = pos
                renderer.menuDurMs = p.duration.coerceAtLeast(0)
                renderer.menuPlaying = p.playWhenReady && p.playbackState == Player.STATE_READY
                val buf = (p.bufferedPosition - pos).coerceAtLeast(0)
                val frames = renderer.frameCount
                val fps = frames - lastWatchFrames
                lastWatchFrames = frames
                val consumed = renderer.consumedFrames
                val vfps = consumed - lastWatchConsumed
                lastWatchConsumed = consumed
                val arrived = renderer.arrivedFrames
                val afps = arrived - lastWatchArrived
                lastWatchArrived = arrived
                // Splitter signals: is the video track still selected, what
                // format does the player think is current, is it playing, and
                // how long since the decoder last delivered a frame batch?
                // (offsetAge growing + vfps 0 = decoder stopped feeding;
                //  drops climbing instead = render-side stall.)
                var vidSel = false
                var vidFmt = "none"
                runCatching {
                    for (g in p.currentTracks.groups) {
                        if (!g.isSelected) continue
                        val f = g.getTrackFormat(0)
                        if (f.sampleMimeType?.startsWith("video/") == true) {
                            vidSel = true
                            vidFmt = "${f.sampleMimeType} ${f.width}x${f.height}"
                        }
                    }
                }
                val offsetAge = System.currentTimeMillis() - lastFrameBatchMs
                FileLog.i(TAG, "watch pos=${pos}ms buf=${buf}ms dur=${p.duration}ms " +
                    "state=$state playWhenReady=${p.playWhenReady} glfps~${fps / 2} vfps~${vfps / 2} afps~${afps / 2} " +
                    "vidSel=$vidSel vidFmt=$vidFmt playing=${p.isPlaying} offsetAge=${offsetAge}ms " +
                    "wh=${renderer.lastWidth}x${renderer.lastHeight} lens=${renderer.lensK1},${renderer.lensK2} " +
                    "proj=${renderer.projection} stereo=${renderer.stereo} zoom=${settings.videoZoom}")
            }
            sensorTick.postDelayed(this, 2000)
        }
    }

    private fun startWatchTick() {
        sensorTick.removeCallbacks(watchTickTask)
        sensorTick.post(watchTickTask)
    }

    override fun onPause() {
        glView.onPause()
        sensors.unregisterListener(this)
        cmpSensor?.let { sensors.unregisterListener(cmpListener) }
        sensors.unregisterListener(envListener)
        stopSensorTick()
        stopTrace()
        player?.pause()
        super.onPause()
    }

    override fun onDestroy() {
        player?.release(); player = null
        if (playIsProxy) playUrl?.let { StreamProxy.unregisterByUrl(it) }
        super.onDestroy()
    }

    override fun onKeyDown(code: Int, e: KeyEvent?): Boolean {
        // Volume keys are deliberately NOT intercepted: they adjust volume.
        // Gaze dwell is the selector; DPAD/enter covers devices that have it.
        if (code == KeyEvent.KEYCODE_DPAD_CENTER || code == KeyEvent.KEYCODE_ENTER) {
            renderer.tapSelect()
            return true
        }
        return super.onKeyDown(code, e)
    }

    private fun hideSystemBars() {
        val ctrl = androidx.core.view.WindowCompat.getInsetsController(window, window.decorView)
        ctrl.hide(androidx.core.view.WindowInsetsCompat.Type.systemBars())
        ctrl.systemBarsBehavior =
            androidx.core.view.WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    // ---------------- browser ----------------
    // Back to the browser, staying in the directory that was open (NOT top
    // level): exiting a video returns to its folder.
    private fun goBrowser() { player?.pause(); enterBrowser() }

    private fun enterBrowser() {
        player?.pause()
        renderer.mode = VrRenderer.Mode.BROWSER
        // Default to the playing file's folder so Files opens where you are.
        // No recenter: the world (video + panel frame) must not move when
        // opening a menu. The panel stays world-locked; look around for it.
        if (playIndex in playQueue.indices) {
            when (val it = playQueue[playIndex]) {
                is PlayItem.Smb -> loc = Loc.Smb(it.connId, it.e.path.substringBeforeLast('\\', ""))
                is PlayItem.Local -> it.f.parentFile?.let { p -> loc = Loc.Local(p) }
                is PlayItem.Saf -> {
                    val cur = loc as? Loc.Saf
                    loc = if (cur != null && cur.treeUri == it.treeUri) Loc.Saf(it.treeUri, it.relPath, cur.label)
                    else {
                        val label = SafFiles.removableVolumes(this).find { v ->
                            SafFiles.grantedTree(this, v.uuid) == it.treeUri
                        }?.desc ?: "SD card"
                        Loc.Saf(it.treeUri, it.relPath, label)
                    }
                }
            }
        } else if (loc !is Loc.Smb && loc !is Loc.Local && loc !is Loc.Saf && loc !is Loc.Root) {
            // Playing file isn't in the folder queue (single-file play) AND
            // we're sitting on a non-file page (settings/shaping/sensors):
            // without this the 📁 button would just re-show that stale page
            // instead of the file browser. Fall back to the server list.
            loc = Loc.Root
        }
        refresh()
    }

    private fun refresh() {
        // Browser panel floats over live video halfway to the play menu
        // when a video exists; centered at horizon otherwise.
        renderer.browserElevDeg = if (player != null) renderer.overlayElevDeg() else 0f
        when (val l = loc) {
            is Loc.Root -> showRoot()
            is Loc.Smb -> showSmb(l.connId, l.path)
            is Loc.Local -> showLocal(l.dir)
            is Loc.Saf -> showSaf(l.treeUri, l.relPath, l.label)
            is Loc.SettingsPage -> showSettings()
            is Loc.ShapingPage -> showShaping()
            is Loc.Sensors -> showSensors()
        }
    }

    private fun pushRows(title: String, status: String, r: List<Row>) {
        rows = r
        renderer.browserTitle = title
        // File pages lead with home + up: pin both above the scroll-up
        // strip so nav stays reachable without scrolling back to the top.
        // Settings/shaping/sensor pages get no strips (pin 0).
        renderer.pinTopRows =
            if (r.size >= 2 && r[0].action == "home:" && r[1].action == "up:") 2
            else 0
        renderer.browserRows = r.map {
            VrRenderer.BrowserRow(it.label, it.meta, it.kind, it.slideKey, it.slideMin, it.slideMax, it.slideVal, it.slideFmt,
                it.segLabels, it.segActions, it.segSelected,
                it.previewMags, it.previewHull, it.previewN, it.previewPos, it.dead)
        }
        txtStatus.text = status
    }

    private fun showRoot() {
        val r = mutableListOf<Row>()
        for (c in connections) {
            r += Row(c.label, "${c.unc} • ${if (c.username.isBlank()) "guest" else c.username}",
                VrRenderer.BrowserRow.FOLDER, action = "smb:${c.id}")
        }
        r += Row("Internal Storage", "phone storage", VrRenderer.BrowserRow.FOLDER, action = "local:")
        for (vr in LocalFiles.volumeRoots(this).filter { !it.isPrimary }) {
            if (LocalFiles.sdBlocked(this, vr.dir))
                r += Row(vr.label, "tap to grant full access",
                    VrRenderer.BrowserRow.FOLDER, action = "grantfull:${vr.dir.absolutePath}")
            else r += Row(vr.label, vr.dir.absolutePath,
                VrRenderer.BrowserRow.FOLDER, action = "local:${vr.dir.absolutePath}")
        }
        for (v in SafFiles.removableVolumes(this)) {
            val granted = SafFiles.grantedTree(this, v.uuid) != null
            if (!LocalFiles.needsFullAccess() && !granted) continue
            r += Row(v.desc, if (granted) "SD card" else "tap to grant access",
                VrRenderer.BrowserRow.FOLDER, action = "safroot:${v.uuid ?: ""}")
        }
        r += Row("Settings", currentOpticsSummary(), VrRenderer.BrowserRow.ACTION, action = "settings:")
        r += Row("Shaping", "warp grids (dome correction)", VrRenderer.BrowserRow.ACTION, action = "shaping:")
        r += Row("Sensor debug", "live raw values", VrRenderer.BrowserRow.ACTION, action = "sensors:")
        pushRows("LimpetVR", if (connections.isEmpty()) "Add a server in the 2D app, or open Internal Storage" else "${connections.size} servers — stare to open", r)
    }

    private fun currentOpticsSummary(): String =
        "${renderer.projection.label} ${renderer.stereo.label} • fov×${String.format("%.2f", settings.fovScale)} • IPD ${settings.ipdMm.toInt()}mm"

    private fun showSmb(connId: String, path: String) {
        val conn = connections.find { it.id == connId }
        if (conn == null) { loc = Loc.Root; refresh(); return }
        pushRows("${conn.host}/${conn.share} /${path.replace('\\', '/')}", "Listing…",
            listOf(Row("Loading…", "", VrRenderer.BrowserRow.FILE)))
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                if (!SmbHolder.manager.isBoundTo(conn.id)) SmbHolder.manager.connect(conn)
                val list = SmbHolder.manager.list(path)
                val r = mutableListOf<Row>()
                r += Row("⌂ (top)", "servers list", VrRenderer.BrowserRow.FOLDER, action = "home:")
                r += Row(".. (up)", if (path.isEmpty()) "back to servers" else "parent folder", VrRenderer.BrowserRow.FOLDER, action = "up:")
                for (e in list) {
                    val kind = if (e.isDir) VrRenderer.BrowserRow.FOLDER
                        else if (e.isVideo()) VrRenderer.BrowserRow.VIDEO else VrRenderer.BrowserRow.FILE
                    val meta = if (e.isDir) "folder" else humanSize(e.size)
                    r += Row(e.name, meta, kind, smb = e)
                }
                withContext(Dispatchers.Main) {
                    if (loc == Loc.Smb(connId, path)) {
                        pushRows("${conn.host}/${conn.share} /${path.replace('\\', '/')}",
                            "${list.size} items • streaming, no download", r)
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "smb list failed", t)
                withContext(Dispatchers.Main) {
                    pushRows("Error", "SMB: ${t.message}",
                        listOf(Row(".. (back)", "${t.message?.take(60)}", VrRenderer.BrowserRow.FOLDER, action = "up:")))
                    toast("SMB: ${t.message}", long = true)
                }
            }
        }
    }

    private fun showSaf(treeUri: String, relPath: String, label: String) {
        val titleRel = if (relPath.isEmpty()) "/" else "/${relPath.replace('/', '/')}"
        pushRows("$label$titleRel", "Listing…",
            listOf(Row("Loading…", "", VrRenderer.BrowserRow.FILE)))
        lifecycleScope.launch(Dispatchers.IO) {
            val kids = try {
                SafFiles.list(this@VrPlayerActivity, android.net.Uri.parse(treeUri), relPath)
            } catch (t: Throwable) { emptyList() }
            val r = mutableListOf<Row>()
            r += Row("⌂ (top)", "servers list", VrRenderer.BrowserRow.FOLDER, action = "home:")
            r += Row(".. (up)", if (relPath.isEmpty()) "back to servers" else "parent folder",
                VrRenderer.BrowserRow.FOLDER, action = "up:")
            for (k in kids) {
                val kind = if (k.isDir) VrRenderer.BrowserRow.FOLDER
                    else if (SafFiles.isVideo(k)) VrRenderer.BrowserRow.VIDEO else VrRenderer.BrowserRow.FILE
                val meta = if (k.isDir) "folder" else humanSize(k.size)
                r += Row(k.name, meta, kind, smb = null, saf = k, safTree = treeUri, safRel = relPath)
            }
            withContext(Dispatchers.Main) {
                if (loc == Loc.Saf(treeUri, relPath, label)) {
                    pushRows("$label$titleRel", "${kids.size} items", r)
                }
            }
        }
    }

    private fun showLocal(dir: File) {
        val (label, rel) = LocalFiles.rootTitle(this, dir)
        pushRows("$label /${dir.name}", "Listing…",
            listOf(Row("Loading…", "", VrRenderer.BrowserRow.FILE)))
        lifecycleScope.launch(Dispatchers.IO) {
            val kids = try { LocalFiles.list(dir) } catch (t: Throwable) { emptyList() }
                val r = mutableListOf<Row>()
                val isRoot = LocalFiles.volumeRoots(this@VrPlayerActivity).any { it.dir == dir }
                r += Row("⌂ (top)", "servers list", VrRenderer.BrowserRow.FOLDER, action = "home:")
                r += Row(".. (up)", if (isRoot) "back to servers" else "parent folder", VrRenderer.BrowserRow.FOLDER, action = "up:")
            for (k in kids) {
                val kind = if (k.isDir) VrRenderer.BrowserRow.FOLDER
                    else VrRenderer.BrowserRow.VIDEO
                r += Row(k.name, if (k.isDir) "folder" else humanSize(k.size), kind, local = k.file)
            }
            withContext(Dispatchers.Main) {
                pushRows("$label $rel", "${kids.size} items", r)
            }
        }
    }

    private fun showSettings() {
        // gaze sliders: dwell anywhere on the bar to jump straight there
        fun slide(label: String, value: String, key: String, min: Float, max: Float, cur: Float,
                  fmt: VrRenderer.SlideFormat = VrRenderer.SlideFormat()) =
            Row(label, value, VrRenderer.BrowserRow.ACTION,
                action = "slide:$key", slideKey = key, slideMin = min, slideMax = max, slideVal = cur,
                slideFmt = fmt)
        val r = mutableListOf(
            Row("Load defaults", "reset calibration to the startup baseline", VrRenderer.BrowserRow.ACTION, action = "set:defaults"),
            Row("Video", renderer.stereo.label, VrRenderer.BrowserRow.ACTION,
                segLabels = listOf("2D", "SBS", "TB"),
                segActions = listOf("setstereo2:MONO", "setstereo2:SBS", "setstereo2:TB"),
                segSelected = when (renderer.stereo) { Stereo.MONO -> 0; Stereo.SBS -> 1; Stereo.TB -> 2 }),
            Row("Screen", renderer.projection.label, VrRenderer.BrowserRow.ACTION,
                segLabels = listOf("Flat", "180", "220", "270", "360"),
                segActions = listOf("setscreen:FLAT", "setscreen:DEG180", "setscreen:DEG220", "setscreen:DEG270", "setscreen:DEG360"),
                segSelected = when (renderer.projection) { Projection.FLAT -> 0; Projection.DEG180 -> 1; Projection.DEG220 -> 2; Projection.DEG270 -> 3; Projection.DEG360 -> 4; else -> -1 }),
            Row("Lens", if (renderer.projection == Projection.FISHEYE) "Fisheye" else "Normal", VrRenderer.BrowserRow.ACTION,
                segLabels = listOf("Normal", "Fisheye"),
                segActions = listOf("setlens:normal", "setlens:fisheye"),
                segSelected = if (renderer.projection == Projection.FISHEYE) 1 else 0),
            slide("Fisheye radius", "${String.format("%.2f", settings.fisheyeRadius)}×", "fishR", 0.5f, 1.5f, settings.fisheyeRadius,
                VrRenderer.SlideFormat("×", 2, 1f, 0f, 0.01f)),
            Row("Mirror right fisheye", if (settings.fisheyeMirrorR) "ON" else "off", VrRenderer.BrowserRow.ACTION, action = "set:fishmirror"),

            slide("FOV scale", "${String.format("%.2f", settings.fovScale)}×", "fov", 0.5f, 1.5f, settings.fovScale,
                VrRenderer.SlideFormat("×", 2, 1f, 0f, 0.05f)),
            slide("Screen size (Flat only)", "${String.format("%.2f", settings.screenSize)}×", "screensize", 0.5f, 10f, settings.screenSize,
                VrRenderer.SlideFormat("×", 2, 1f, 0f, 0.25f)),
            slide("Screen curve (Flat only)", "${(settings.screenCurve * 100).toInt()}%", "curve", 0f, 1f, settings.screenCurve,
                VrRenderer.SlideFormat("%", 0, 100f, 0f, 0.05f)),
            slide("Video size", "${String.format("%.2f", settings.videoZoom)}×", "zoom", 0.1f, 20f, settings.videoZoom,
                VrRenderer.SlideFormat("×", 2, 1f, 0f, 0.05f)),
            Row("", "", VrRenderer.BrowserRow.FILE, dead = true),
            slide("Eye separation", "${settings.ipdMm.toInt()} mm", "ipd", 40f, 80f, settings.ipdMm,
                VrRenderer.SlideFormat(" mm", 0, 1f, 0f, 1f)),
            slide("Convergence", "${String.format("%+.3f", settings.convTrim)}", "convtrim", -0.15f, 0.15f, settings.convTrim,
                VrRenderer.SlideFormat("", 3, 1f, 0f, 0.005f)),
            slide("Panel distance", "${String.format("%.1f", settings.panelDistM)} m", "panel", 1.2f, 5f, settings.panelDistM,
                VrRenderer.SlideFormat(" m", 1, 1f, 0f, 0.1f)),
            Row("Gaze tooltip", if (settings.enableTooltip) "ON" else "off", VrRenderer.BrowserRow.ACTION, action = "set:tooltip"),
            Row("Lens distortion", if (settings.disableDist) "off" else "ON", VrRenderer.BrowserRow.ACTION, action = "set:distort"),
            Row("Pano mesh", if (settings.panoQuality == "vertexhq") "dense" else "normal", VrRenderer.BrowserRow.ACTION,
                segLabels = listOf("normal", "dense"),
                segActions = listOf("setpano:vertex", "setpano:vertexhq"),
                segSelected = if (settings.panoQuality == "vertexhq") 1 else 0),
        )
        pushRows("Video settings", "stare at a bar position to jump there", r)
    }

    private fun showShaping() {
        fun slide(label: String, value: String, key: String, min: Float, max: Float, cur: Float,
                  fmt: VrRenderer.SlideFormat = VrRenderer.SlideFormat()) =
            Row(label, value, VrRenderer.BrowserRow.ACTION,
                action = "slide:$key", slideKey = key, slideMin = min, slideMax = max, slideVal = cur,
                slideFmt = fmt)
        val r = mutableListOf<Row>()
        // Global lens pre-warp first (above the Combined mesh): drag k1/k2
        // to 0 and the warp visibly vanishes, which proves the coefficients
        // are applied. Same keys as before, so dwell/nudge handlers work.
        r += slide("Lens k1", String.format("%.2f", settings.lensK1), "lensK1", 0f, 1f, settings.lensK1,
            VrRenderer.SlideFormat("", 2, 1f, 0f, 0.01f))
        r += slide("Lens k2", String.format("%.2f", settings.lensK2), "lensK2", 0f, 1f, settings.lensK2,
            VrRenderer.SlideFormat("", 2, 1f, 0f, 0.01f))
        r += slide("Lens strength", String.format("%.2f×", settings.lensStrength), "lensStrength", 0f, 3f, settings.lensStrength,
            VrRenderer.SlideFormat("×", 2, 1f, 0f, 0.05f))
        // Combined preview of the averaged transform (not actionable).
        val avg = averagedShapeOffsets()
        if (avg != null) {
            val (ex, ey, n) = avg
            var mmax = 0f
            val mags = FloatArray(ex.size)
            for (j in ex.indices) {
                val m = kotlin.math.hypot(ex[j], ey[j])
                mags[j] = m; if (m > mmax) mmax = m
            }
            val norm = if (mmax > 1e-9f) FloatArray(ex.size) { mags[it] / mmax } else FloatArray(ex.size)
            r += Row("Combined", "averaged transform", VrRenderer.BrowserRow.FILE,
                previewMags = norm, previewHull = convexHull(ex, ey, mags, n), previewN = n,
                previewPos = previewPos(ex, ey, n))
        } else {
            r += Row("Combined", "nothing enabled — identity", VrRenderer.BrowserRow.FILE)
        }
        for (s in shapingShapes) {
            val on = settings.shapeEnabled(s.id)
            val w = settings.shapeWeight(s.id)
            r += Row(s.label, if (on) "ON • ${w.toInt()}%" else "off",
                VrRenderer.BrowserRow.ACTION, action = "toggleshape:${s.id}",
                previewMags = s.mags, previewHull = s.hull, previewN = s.n,
                previewPos = previewPos(s.ox, s.oy, s.n))
            r += slide("${s.label} weight", "${w.toInt()}%", "shapeWeight-${s.id}", 0f, 100f, w,
                VrRenderer.SlideFormat("%", 0, 1f, 0f, 1f))
        }
        pushRows("Shaping", "dwell a shape to toggle • dwell a bar to set weight", r)
    }

    /** Live raw sensor readout. Fixed row count; navigation is frozen here
     *  (activateRow ignores everything but Back) so staring can't fire. */
    private fun showSensors() {
        fun f(v: Float) = if (v >= 0) "+${String.format("%.2f", v)}" else String.format("%.2f", v)
        fun ypr(m: FloatArray?): IntArray {
            if (m == null) return intArrayOf(999, 999, 999)
            val o = FloatArray(3)
            SensorManager.getOrientation(m, o)
            return intArrayOf(
                Math.toDegrees(o[0].toDouble()).toInt(),
                Math.toDegrees(o[1].toDouble()).toInt(),
                Math.toDegrees(o[2].toDouble()).toInt()
            )
        }
        fun yprOf(m: FloatArray?): String {
            val d = ypr(m)
            return if (d[0] > 900) "n/a" else "y${d[0]} p${d[1]} r${d[2]}"
        }
        val g = lastGyro
        val ac = lastAccel
        val mg = lastMag
        val gmag = kotlin.math.sqrt((g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).toDouble())
        val amag = kotlin.math.sqrt((ac[0]*ac[0] + ac[1]*ac[1] + ac[2]*ac[2]).toDouble())
        val mmag = kotlin.math.sqrt((mg[0]*mg[0] + mg[1]*mg[1] + mg[2]*mg[2]).toDouble())
        // Turn detector (gimbal-free): gyro-integrated yaw about accel-up vs
        // rendered-forward yaw, both from page-open baselines. A real head
        // turn MUST move both, opposite signs (head-R ⇒ image-L), ~equal
        // magnitude. Head tilt moves NEITHER (correct: no pan on tilt).
        // (Euler yaw rows above are gimbal-unreliable in-viewer; this row
        // and the trace file are the ground truth.)
        val effM = renderer.effCopy()
        val rfx = -effM[2]; val rfz = -effM[10]
        // render yaw measured from recenter-forward; baseline at page open
        val renderYawNow = Math.toDegrees(kotlin.math.atan2(rfx.toDouble(), -rfz.toDouble()))
        if (turnBaseG == null || turnBaseR == null) {
            turnBaseG = yawGyroRelDeg; turnBaseR = renderYawNow
        }
        fun wrap(d: Double): Double {
            var x = d % 360.0
            if (x > 180) x -= 360
            if (x < -180) x += 360
            return x
        }
        val turnRow = run {
            val dg = wrap(yawGyroRelDeg - turnBaseG!!)
            val dr = wrap(renderYawNow - turnBaseR!!)
            val ok = (kotlin.math.abs(dg) < 3 && kotlin.math.abs(dr) < 3) ||
                (kotlin.math.abs(dg) > 8 && ((dg > 0) != (dr > 0)) &&
                    kotlin.math.abs(kotlin.math.abs(dg) - kotlin.math.abs(dr)) <
                        kotlin.math.max(12.0, kotlin.math.abs(dg) * 0.5))
            Row("TURN gyro=${if (dg >= 0) "+" else ""}${dg.toInt()} img=${if (dr >= 0) "+" else ""}${dr.toInt()} ${if (ok) "OK" else "WRONG"}",
                "turn: opposite, ~equal • tilt: both ~0", VrRenderer.BrowserRow.ACTION)
        }
        val r = listOf(
            Row("GYRO ${f(g[0])} ${f(g[1])} ${f(g[2])}", "rad/s — turn head L/R, one axis must swing", VrRenderer.BrowserRow.ACTION),
            Row("GYROmag ${String.format("%.2f", gmag)}", "still=~0, turning=spikes", VrRenderer.BrowserRow.ACTION),
            Row("ACC ${f(ac[0])} ${f(ac[1])} ${f(ac[2])}", "|g|=${String.format("%.1f", amag)} — still=~9.8", VrRenderer.BrowserRow.ACTION),
            Row("MAG ${f(mg[0])} ${f(mg[1])} ${f(mg[2])}", "|m|=${String.format("%.0f", mmag)} — still~=const", VrRenderer.BrowserRow.ACTION),
            Row("GAMErv ${yprOf(lastTrackM)}", if (useGameRv) "tracking source" else "compare source", VrRenderer.BrowserRow.ACTION),
            Row("FULLrv ${yprOf(cmpListener.lastRaw)}", if (useGameRv) "compare source" else "tracking source", VrRenderer.BrowserRow.ACTION),
            Row("RENDER ${yprOf(renderer.effCopy())}", "snaps=${renderer.snapCount} kept=${renderer.keptFrames} — must follow GAMErv", VrRenderer.BrowserRow.ACTION),
            turnRow,
            Row("PEAK gyro ${String.format("%.2f", peakGyro[0])}/${String.format("%.2f", peakGyro[1])}/${String.format("%.2f", peakGyro[2])}",
                "max rate per axis since page opened", VrRenderer.BrowserRow.ACTION),
            Row("PEAK gaze ${String.format("%.0f", peakGazeDeg)}° $peakGazeDir",
                "max image deflection since page opened", VrRenderer.BrowserRow.ACTION),
        )
        pushRows("Sensors — turn head L/R", "nav frozen here • X exits", r)
    }

    private fun logSensors() {
        val g = lastGyro; val ac = lastAccel; val mg = lastMag
        val e = renderer.effCopy()
        val eo = FloatArray(3); SensorManager.getOrientation(e, eo)
        val ed = eo.map { Math.toDegrees(it.toDouble()).toInt() }
        Log.i("LimpetVR-sensors",
            "gyro=${g[0].fmt()}|${g[1].fmt()}|${g[2].fmt()} " +
            "acc=${ac[0].fmt()}|${ac[1].fmt()}|${ac[2].fmt()} " +
            "mag=${mg[0].fmt()}|${mg[1].fmt()}|${mg[2].fmt()} " +
            "render=y${ed[0]}p${ed[1]}r${ed[2]} snaps=${renderer.snapCount}")
    }

    private fun Float.fmt(): String = String.format("%.2f", this)

    /** Dual toast: 2D system toast plus the in-headset center toast
     *  (system toasts are unreadable in VR). */
    private fun toast(msg: String, long: Boolean = false) {
        Toast.makeText(this, msg, if (long) Toast.LENGTH_LONG else Toast.LENGTH_SHORT).show()
        renderer.showToast(msg)
    }

    /** X close button (title bar): resume video if one exists, else server list.
     *  Never recenters: closing must not move the world. */
    private fun closeOverlay() {
        if (player != null) {
            settingsFromVideo = false
            renderer.mode = VrRenderer.Mode.VIDEO
        } else {
            settingsFromVideo = false
            loc = Loc.Root
            refresh()
        }
    }

    private fun activateRow(idx: Int, frac: Float? = null) {
        if (idx == -10) { if (renderer.mode == VrRenderer.Mode.BROWSER) closeOverlay(); return }
        if (idx !in rows.indices || renderer.mode != VrRenderer.Mode.BROWSER) return
        // Debug page: frozen except X, so staring at numbers is safe.
        if (loc == Loc.Sensors) {
            if (rows[idx].action == "up:") handleAction("up:")
            return
        }
        val row = rows[idx]
        // SAF (SD card) entries: folders navigate, videos play direct.
        row.saf?.let { e ->
            if (loc is Loc.Saf) {
                val l = loc as Loc.Saf
                if (e.isDir) {
                    val child = if (l.relPath.isEmpty()) e.name else "${l.relPath}/${e.name}"
                    loc = Loc.Saf(l.treeUri, child, l.label)
                    refresh()
                } else if (SafFiles.isVideo(e)) playSafFile(e.uri.toString(), e.name, l.treeUri, l.relPath)
                else toast("Not a playable video")
            }
            return
        }
        // segmented button row: horizontal gaze fraction picks the segment
        if (row.segActions.isNotEmpty()) {
            if (frac == null) return // DPAD tap carries no position; gaze only
            // frac spans the full panel; buttons span x 20..1004 of TEX 1024
            val fx = ((frac * 1024f - 20f) / 984f).coerceIn(0f, 0.999f)
            val seg = (fx * row.segActions.size).toInt().coerceIn(0, row.segActions.size - 1)
            FileLog.i("LimpetVR-browser", "seg fire: row=${row.label} seg=$seg action=${row.segActions[seg]}")
            handleAction(row.segActions[seg]); return
        }
        // gaze slider: one dwell at fraction u sets the value directly
        if (row.slideKey != null && frac != null) {
            handleSlide(row.slideKey, frac)
            return
        }
        val act = row.action
        if (act != null) { handleAction(act); return }
        when (val l = loc) {
            is Loc.Smb -> row.smb?.let { e ->
                if (e.isDir) { loc = Loc.Smb(l.connId, e.path); refresh() }
                else if (e.isVideo()) playSmbFile(e, l.connId)
                else toast("Not a playable video")
            }
            is Loc.Local -> row.local?.let { f ->
                if (f.isDirectory) { loc = Loc.Local(f); refresh() }
                else playLocalFile(f)
            }
            else -> {}
        }
    }

    /** Gaze-slider set: fraction across the row maps to the key's range,
     *  snapped to its grid. One dwell reaches any value. */
    private fun handleSlide(key: String, frac: Float) {
        val f = frac.coerceIn(0f, 1f)
        FileLog.i("LimpetVR-browser", "slide key=$key frac=$f")
        when (key) {
            "fov" -> settings.fovScale = ((0.5f + f * 1f) * 20f).roundToInt() / 20f
            "screensize" -> settings.screenSize = ((0.5f + f * 9.5f) * 4f).roundToInt() / 4f
            "curve" -> settings.screenCurve = ((f * 20f).roundToInt() / 20f).coerceIn(0f, 1f)
            "zoom" -> settings.videoZoom = ((0.1f + f * 19.9f) * 20f).roundToInt() / 20f
            "ipd" -> settings.ipdMm = (40f + f * 40f).roundToInt().toFloat().coerceIn(40f, 80f)
            "convtrim" -> settings.convTrim = (((f * 0.3f - 0.15f) / 0.005f).roundToInt() * 0.005f).coerceIn(-0.15f, 0.15f)
            "panel" -> settings.panelDistM = ((1.2f + f * 3.8f) * 10f).roundToInt() / 10f
            "lensK1" -> settings.lensK1 = ((f * 100f).roundToInt() / 100f).coerceIn(0f, 1f)
            "lensK2" -> settings.lensK2 = ((f * 100f).roundToInt() / 100f).coerceIn(0f, 1f)
            "fishR" -> settings.fisheyeRadius = ((0.5f + f * 1f) * 100f).roundToInt() / 100f
            "lensStrength" -> settings.lensStrength = ((f * 3f * 20f).roundToInt() / 20f).coerceIn(0f, 3f)

            "dwell" -> settings.dwellMs = ((400f + f * 3600f) / 100f).roundToInt() * 100L
            else -> {
                // per-shape weight sliders: slideKey "shapeWeight-<slug>"
                if (key.startsWith("shapeWeight-")) {
                    val id = key.removePrefix("shapeWeight-")
                    if (shapingShapes.any { it.id == id })
                        settings.setShapeWeight(id, (f * 100f).roundToInt().toFloat().coerceIn(0f, 100f))
                }
            }
        }
        applyOptics(); refresh()
    }

    private fun handleAction(act: String) {
        when {
            act == "home:" -> { loc = Loc.Root; refresh() }
            act == "up:" -> {
                loc = when (val l = loc) {
                    is Loc.Smb -> if (l.path.isEmpty()) Loc.Root else Loc.Smb(l.connId, l.path.substringBeforeLast('\\', ""))
                    is Loc.Local -> {
                        val isRoot = LocalFiles.volumeRoots(this).any { it.dir == l.dir }
                        if (isRoot || l.dir.parentFile == null) Loc.Root else Loc.Local(l.dir.parentFile!!)
                    }
                    is Loc.Saf -> if (l.relPath.isEmpty()) Loc.Root
                        else Loc.Saf(l.treeUri, l.relPath.substringBeforeLast('/', ""), l.label)
                    else -> Loc.Root
                }
                refresh()
            }
            act == "settings:" -> { settingsFromVideo = false; loc = Loc.SettingsPage; refresh() }
            act == "shaping:" -> { settingsFromVideo = false; loc = Loc.ShapingPage; refresh() }
            act == "shapingback:" -> {
                if (settingsFromVideo && player != null) {
                    settingsFromVideo = false
                    renderer.mode = VrRenderer.Mode.VIDEO
                } else {
                    settingsFromVideo = false
                    loc = Loc.Root
                    refresh()
                }
            }
            act.startsWith("toggleshape:") -> {
                val id = act.removePrefix("toggleshape:")
                if (shapingShapes.any { it.id == id }) {
                    settings.setShapeEnabled(id, !settings.shapeEnabled(id))
                    applyOptics(); refresh()
                }
            }
            act == "settingsback:" -> {
                // Back out of settings: resume the video if we came from it,
                // otherwise behave like going up to the server list.
                if (settingsFromVideo && player != null) {
                    settingsFromVideo = false
                    renderer.mode = VrRenderer.Mode.VIDEO
                } else {
                    settingsFromVideo = false
                    loc = Loc.Root
                    refresh()
                }
            }
            act.startsWith("setstereo2:") -> {
                val s = runCatching { Stereo.valueOf(act.removePrefix("setstereo2:")) }.getOrDefault(Stereo.SBS)
                settings.stereo = s; renderer.stereo = s; refresh()
            }
            act.startsWith("setscreen:") -> {
                val p = runCatching { Projection.valueOf(act.removePrefix("setscreen:")) }.getOrDefault(Projection.DEG180)
                settings.screenProj = p; settings.projection = p; renderer.projection = p; syncGvr(); refresh()
            }
            act.startsWith("safroot:") -> {
                val uuid = act.removePrefix("safroot:").ifEmpty { null }
                val tree = SafFiles.grantedTree(this, uuid)
                if (tree != null) {
                    val label = SafFiles.removableVolumes(this).find { it.uuid == uuid }?.desc ?: "SD card"
                    loc = Loc.Saf(tree, "", label); refresh()
                } else {
                    pendingSafVolume = uuid
                    try { safPicker.launch(null) }
                    catch (_: Exception) {
                        Toast.makeText(this, "No folder picker available", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            act.startsWith("setlens:") -> {
                if (act.removePrefix("setlens:") == "fisheye") {
                    settings.projection = Projection.FISHEYE; renderer.projection = Projection.FISHEYE
                } else {
                    settings.projection = settings.screenProj; renderer.projection = settings.screenProj
                }
                syncGvr()
                refresh()
            }
            act == "sensors:" -> { loc = Loc.Sensors; yawBaseG = null; yawBaseR = null; yawBaseC = null; turnBaseG = null; turnBaseR = null; resetPeaks(); refresh() }
            act.startsWith("smb:") -> { loc = Loc.Smb(act.removePrefix("smb:"), ""); refresh() }
            act.startsWith("local:") -> {
                if (LocalFiles.needsPermission(this)) {
                    toast("Allow videos permission in the 2D app first", long = true)
                } else {
                    val d = File(act.removePrefix("local:"))
                    if (d.path.isEmpty()) loc = Loc.Local(LocalFiles.externalRoot())
                    else if (LocalFiles.sdBlocked(this, d)) {
                        toast("Allow all-files access in the 2D app first", long = true)
                    } else loc = Loc.Local(d)
                }
                refresh()
            }
            act.startsWith("grantfull:") -> {
                // drops to the system Settings toggle; onResume re-checks
                LocalFiles.requestFullAccess(this)
                refresh()
            }
            act == "set:proj" -> { cycleProjection(); refresh() }
            act == "set:stereo" -> {
                renderer.stereo = when (renderer.stereo) { Stereo.MONO -> Stereo.SBS; Stereo.SBS -> Stereo.TB; Stereo.TB -> Stereo.MONO }
                settings.stereo = renderer.stereo; refresh()
            }
            act == "set:swap" -> { settings.swapEyes = !settings.swapEyes; applyOptics(); refresh() }
            act == "set:pin" -> { settings.pinVideo = !settings.pinVideo; applyOptics(); refresh() }
            act == "set:fishmirror" -> { settings.fisheyeMirrorR = !settings.fisheyeMirrorR; applyOptics(); refresh() }
            act == "set:tooltip" -> { settings.enableTooltip = !settings.enableTooltip; applyOptics(); refresh() }
            act == "set:distort" -> { settings.disableDist = !settings.disableDist; applyOptics(); refresh() }
            act.startsWith("setpano:") -> {
                settings.panoQuality = if (act.removePrefix("setpano:") == "vertexhq") "vertexhq" else "vertex"
                applyOptics(); refresh()
            }
            act == "set:defaults" -> {
                settings.fireBaseline(System.currentTimeMillis())
                applyOptics(); refresh()
                renderer.showToast("Defaults loaded", 2000L)
            }
            act.startsWith("adj:") -> {
                val parts = act.split(":")
                val key = parts[1]; val dir = if (parts[2] == "+") 1 else -1
                when (key) {
                    "fov" -> settings.fovScale = (settings.fovScale + dir * 0.05f).coerceIn(0.5f, 1.5f)
                    "screensize" -> settings.screenSize = (settings.screenSize + dir * 0.25f).coerceIn(0.5f, 10f)
                    "curve" -> settings.screenCurve = (settings.screenCurve + dir * 0.05f).coerceIn(0f, 1f)
                    "ipd" -> settings.ipdMm = (settings.ipdMm + dir * 1f).coerceIn(40f, 80f)
                    "zoom" -> settings.videoZoom = (settings.videoZoom * if (dir > 0) 1.25f else 0.8f).coerceIn(0.1f, 20f)
                    "convtrim" -> settings.convTrim = (settings.convTrim + dir * 0.005f).coerceIn(-0.15f, 0.15f)

                    "panel" -> settings.panelDistM = (settings.panelDistM + dir * 0.2f).coerceIn(1.2f, 5f)
                    "lensK1" -> settings.lensK1 = (settings.lensK1 + dir * 0.02f).coerceIn(0f, 1f)
                    "lensK2" -> settings.lensK2 = (settings.lensK2 + dir * 0.02f).coerceIn(0f, 1f)
                    "fishR" -> settings.fisheyeRadius = (settings.fisheyeRadius + dir * 0.01f).coerceIn(0.5f, 1.5f)
                    "lensStrength" -> settings.lensStrength = (settings.lensStrength + dir * 0.1f).coerceIn(0f, 3f)
                    "dwell" -> settings.dwellMs = (settings.dwellMs + dir * 250).coerceIn(400L, 4000L)
                    else -> {
                        if (key.startsWith("shapeWeight-")) {
                            val id = key.removePrefix("shapeWeight-")
                            if (shapingShapes.any { it.id == id })
                                settings.setShapeWeight(id, (settings.shapeWeight(id) + dir * 5f).coerceIn(0f, 100f))
                        }
                    }
                }
                applyOptics(); refresh()
            }
        }
    }

    // ---------------- playback ----------------
    private fun playSmbFile(e: SmbEntry, connId: String, recenter: Boolean = true) {
        txtStatus.text = "Opening ${e.name}… (streaming, no download)"
        Log.i(TAG, "open smb: ${e.path}")
        // Remember the folder: re-entering VR (or the 2D app) must land
        // back in the directory the video was started from, not top level.
        SessionMemory.lastConnectionId = "smb:$connId"
        SessionMemory.lastPath = e.path.substringBeforeLast('\\', "")
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val probe = SmbHolder.manager.openRead(e.path)
                val size = probe.size
                probe.close()
                Log.i(TAG, "smb probed, size=$size")
                val url = StreamProxy.register(
                    opener = { kotlinx.coroutines.runBlocking { SmbHolder.manager.openRead(e.path) } },
                    size = size,
                    displayName = e.name
                )
                if (playIsProxy) playUrl?.let { StreamProxy.unregisterByUrl(it) }
                playIsProxy = true; playUrl = url
                withContext(Dispatchers.Main) { captureQueueSmb(e, connId); startPlayback(url, e.name, recenter) }
            } catch (t: Throwable) {
                Log.e(TAG, "smb open failed", t)
                withContext(Dispatchers.Main) {
                    txtStatus.text = "Open failed: ${t.message}"
                    toast("Open failed: ${t.message}", long = true)
                }
            }
        }
    }

    private fun playLocalFile(f: File, recenter: Boolean = true) {
        if (playIsProxy) playUrl?.let { StreamProxy.unregisterByUrl(it) }
        playIsProxy = false
        playUrl = Uri.fromFile(f).toString()
        SessionMemory.lastConnectionId = "local:${f.parentFile?.absolutePath ?: ""}"
        SessionMemory.lastPath = ""
        captureQueue(f)
        startPlayback(playUrl!!, f.name, recenter)
    }

    /** SD-card video via SAF document Uri (ExoPlayer plays it directly). */
    private fun playSafFile(uri: String, name: String, treeUri: String, relPath: String, recenter: Boolean = true) {
        if (playIsProxy) playUrl?.let { StreamProxy.unregisterByUrl(it) }
        playIsProxy = false
        playUrl = uri
        SessionMemory.lastConnectionId = "saf:$treeUri"
        SessionMemory.lastPath = relPath
        captureQueueSaf(uri, treeUri, relPath)
        startPlayback(playUrl!!, name, recenter)
    }

    private fun captureQueueSaf(currentUri: String, treeUri: String, relPath: String) {
        val vids = rows.mapNotNull { r ->
            val e = r.saf
            if (r.kind == VrRenderer.BrowserRow.VIDEO && e != null && !e.isDir)
                PlayItem.Saf(e.uri.toString(), e.name, treeUri, relPath) else null
        }
        if (vids.isNotEmpty()) {
            playQueue = vids
            playIndex = vids.indexOfFirst { it.uri == currentUri }
        } else {
            playIndex = playQueue.indexOfFirst { (it as? PlayItem.Saf)?.uri == currentUri }
        }
    }

    // ---------------- play menu ----------------
    /** Sibling videos of the playing file, for prev/next. Captured at play
     *  time from the folder listing (queue = folder as it was opened). */
    private sealed class PlayItem {
        data class Smb(val e: SmbEntry, val connId: String) : PlayItem()
        data class Local(val f: File) : PlayItem()
        data class Saf(val uri: String, val name: String, val treeUri: String, val relPath: String) : PlayItem()
    }
    private var playQueue: List<PlayItem> = emptyList()
    private var playIndex = -1

    private fun captureQueue(local: File) {
        val sibs = local.parentFile?.listFiles()
            ?.filter { it.isFile && LocalFiles.isVideoFile(it) }
            ?.sortedBy { it.name.lowercase() } ?: emptyList()
        if (sibs.isNotEmpty()) {
            playQueue = sibs.map { PlayItem.Local(it) }
            playIndex = sibs.indexOfFirst { it.absolutePath == local.absolutePath }
        } else {
            // No listable dir (stepped from a 2D-started queue): keep the
            // existing queue, relocate the new index inside it.
            playIndex = playQueue.indexOfFirst {
                (it as? PlayItem.Local)?.f?.absolutePath == local.absolutePath
            }
        }
    }

    private fun captureQueueSmb(current: SmbEntry, connId: String) {
        val vids = rows.mapNotNull { r ->
            val e = r.smb
            if (r.kind == VrRenderer.BrowserRow.VIDEO && e != null && !e.isDir) PlayItem.Smb(e, connId) else null
        }
        if (vids.isNotEmpty()) {
            playQueue = vids
            playIndex = vids.indexOfFirst { it.e.path == current.path }
        } else {
            // Browser holds no listing (2D-started playback, or stepped
            // mid-video): keep the queue, relocate the new index inside it.
            playIndex = playQueue.indexOfFirst {
                (it as? PlayItem.Smb)?.e?.path == current.path
            }
        }
    }

    /** The step ramp for zoom+/−: s(z) = 0.12·z + 0.6 — 0.61 at the zoom
     *  minimum (0.1) up to 3.0 at the maximum (20), one smooth function
     *  across the whole range (the old ×1.5 was +0.05 at the bottom and
     *  +5 … +27 at the top). */
    private val zoomSlope = 0.12f
    /** s(z) = zoomSlope·(z + zoomOff): the offset that puts the ramp's
     *  fixed point at −zoomOff (0.6/0.12 = 5.0). */
    private val zoomOff = 0.6f / zoomSlope
    /** e^zoomSlope — one press = one unit of the flow dz/dt = s(z). */
    private val zoomK = Math.exp(zoomSlope.toDouble()).toFloat()

    /** Zoom+/− on the play panel (menu 8/9). A press moves z by one unit
     *  of that flow, in closed form z ← (z + zoomOff)·e^{±zoomSlope} −
     *  zoomOff, so the rate at every point is exactly s(z) AND the two
     *  directions are algebraic inverses: up then down returns to the same
     *  value, where plain Euler (z ± s(z)) drifts ~0.05 per pair. No flash
     *  on a change — the picture is the feedback; only the ends flash,
     *  where nothing would move. */
    private fun zoomStep(dir: Int) {
        val z = settings.videoZoom
        val t = if (dir > 0) (z + zoomOff) * zoomK - zoomOff
                else (z + zoomOff) / zoomK - zoomOff
        val target = t.coerceIn(0.1f, 20f)
        if (Math.abs(target - z) < 1e-4f) {
            renderer.flashMenu(if (dir > 0) "Zoom max" else "Zoom min")
            return
        }
        settings.videoZoom = target
        applyOptics()
    }

    private fun handleMenuEvent(e: VrRenderer.MenuEvent) {
        val p = player ?: return
        when (e) {
            is VrRenderer.MenuEvent.Press -> when (e.idx) {
                0 -> {
                    // 3D settings page over the video (player keeps running,
                    // so zoom/FOV/type changes preview live on return).
                    // No recenter: opening must not move the world.
                    settingsFromVideo = true
                    loc = Loc.SettingsPage
                    renderer.mode = VrRenderer.Mode.BROWSER
                    refresh()
                }
                1 -> {
                    // 3D shaping page over the video (same resume behaviour).
                    settingsFromVideo = true
                    loc = Loc.ShapingPage
                    renderer.mode = VrRenderer.Mode.BROWSER
                    refresh()
                }
                2 -> enterBrowser() // 3D file browser, opens the playing file's folder
                3 -> stepQueue(-1)
                4 -> p.seekTo((p.currentPosition - settings.skipSecs * 1000).coerceAtLeast(0))
                5 -> p.playWhenReady = !p.playWhenReady
                6 -> {
                    val d = p.duration.coerceAtLeast(0)
                    p.seekTo((p.currentPosition + settings.skipSecs * 1000).coerceAtMost(d))
                }
                7 -> stepQueue(1)
                8 -> zoomStep(+1)
                9 -> zoomStep(-1)
                10 -> audioManager().adjustStreamVolume(
                    android.media.AudioManager.STREAM_MUSIC,
                    android.media.AudioManager.ADJUST_RAISE,
                    android.media.AudioManager.FLAG_SHOW_UI)
                11 -> audioManager().adjustStreamVolume(
                    android.media.AudioManager.STREAM_MUSIC,
                    android.media.AudioManager.ADJUST_LOWER,
                    android.media.AudioManager.FLAG_SHOW_UI)
                12 -> {
                    // ⇅ flip: sweep the menu top<->bottom, then auto-hide as
                    // the gaze leaves the activation zone (existing close band).
                    renderer.menuToggleSide()
                    settings.menuTop = renderer.menuSideUp
                }
                13 -> renderer.requestAim() // menu closes, blue aim arms
                14 -> { // FOV scale -0.05 (panoramas narrow their frustum)
                    settings.fovScale = (settings.fovScale - 0.05f).coerceIn(0.5f, 1.5f)
                    applyOptics()
                }
                15 -> { // FOV scale +0.05
                    settings.fovScale = (settings.fovScale + 0.05f).coerceIn(0.5f, 1.5f)
                    applyOptics()
                }
            }
            is VrRenderer.MenuEvent.Seek -> {
                val d = p.duration.coerceAtLeast(0)
                if (d > 0) {
                    // optimistic position: bar jumps before the seek lands
                    val target = (e.frac.coerceIn(0f, 1f) * d).toLong()
                    renderer.menuPosMs = target
                    p.seekTo(target)
                }
            }
        }
    }

    private fun audioManager(): android.media.AudioManager =
        getSystemService(Context.AUDIO_SERVICE) as android.media.AudioManager

    private fun stepQueue(dir: Int) {
        if (playQueue.isEmpty() || playIndex < 0) {
            renderer.flashMenu("No folder queue")
            return
        }
        // Wraps around the ends: next on the last file plays the first,
        // prev on the first plays the last (floorMod for the -1 side).
        val n = Math.floorMod(playIndex + dir, playQueue.size)
        if (n == playIndex) {
            renderer.flashMenu("Only one video")
            return
        }
        playIndex = n
        when (val it = playQueue[n]) {
            is PlayItem.Smb -> playSmbFile(it.e, it.connId, recenter = false)
            is PlayItem.Local -> playLocalFile(it.f, recenter = false)
            is PlayItem.Saf -> playSafFile(it.uri, it.name, it.treeUri, it.relPath, recenter = false)
        }
    }

    private var playerSwDecode: Boolean? = null
    private fun startPlayback(url: String, name: String, recenter: Boolean = true) {
        renderer.mode = VrRenderer.Mode.VIDEO
        renderer.menuTitle = name
        // Queue steps keep the current center; fresh plays center on the gaze.
        if (recenter) renderer.resetBasis("video")
        // Decoder preference is baked at player build time; rebuild if the
        // user flipped it since (applies to videos opened after changing).
        if (player == null || playerSwDecode != settings.softwareDecode) {
            try { player?.release() } catch (_: Exception) {}
            player = null
            playerSwDecode = settings.softwareDecode
            attachedSurfaceGen = -1 // force surface attach for the new player
        }
        if (player == null) {
            // Buffering sized for the 256MB Java heap: the old 300MB/180s
            // target could retain more samples than the heap holds (8K
            // HEVC fills it in seconds on fast storage) and OOM-killed
            // playback — a consistent crash other players don't have.
            // 60s covers SMB jitter; the 90MB byte backstop keeps total
            // retention inside the heap with headroom. Time thresholds
            // still take priority so audio-front-loaded files can't
            // strand the video track.
            val loadControl = androidx.media3.exoplayer.DefaultLoadControl.Builder()
                .setBufferDurationsMs(15_000, 60_000, 2_500, 5_000)
                .setTargetBufferBytes(90 * 1024 * 1024)
                .setPrioritizeTimeOverSizeThresholds(true)
                .build()
            val preferSw = settings.softwareDecode
            val renderers = androidx.media3.exoplayer.DefaultRenderersFactory(this)
                .setMediaCodecSelector(
                    androidx.media3.exoplayer.mediacodec.MediaCodecSelector { mimeType, requiresSecure, requiresTunnel ->
                        val all = androidx.media3.exoplayer.mediacodec.MediaCodecSelector.DEFAULT
                            .getDecoderInfos(mimeType, requiresSecure, requiresTunnel)
                        val sorted = if (preferSw) {
                            all.sortedBy { info ->
                                val n = info.name
                                if (n.startsWith("OMX.google.") || n.startsWith("c2.android.")) 0 else 1
                            }
                        } else all
                        FileLog.i(TAG, "decoder pick: ${sorted.firstOrNull()?.name} for $mimeType (sw=$preferSw)")
                        sorted
                    })
            player = ExoPlayer.Builder(this, renderers)
                .setLoadControl(loadControl)
                .build().also { exo ->
                exo.playWhenReady = true
                // Engine-verbose internals to logcat (pull with logcat -d):
                // decoder init/release, formats, rendered/dropped counters.
                exo.addAnalyticsListener(androidx.media3.exoplayer.util.EventLogger())
                exo.addListener(object : Player.Listener {
                    override fun onPlayerError(error: PlaybackException) {
                        FileLog.e(TAG, "player error: ${error.message}", error)
                        txtStatus.text = "Playback error: ${error.errorCodeName} — ${error.message?.take(120)}"
                        toast("Playback error: ${error.message}", long = true)
                    }
                    override fun onTracksChanged(tracks: Tracks) {
                        // Codec/resolution/bitrate once known — rules out
                        // decoder-limit issues (e.g. exotic codec, 8K, AV1).
                        val sb = StringBuilder("tracks")
                        for (g in tracks.groups) {
                            if (!g.isSelected) continue
                            val f = g.getTrackFormat(0)
                            sb.append(" [${f.sampleMimeType} ${f.width}x${f.height}" +
                                " br=${f.bitrate} codecs=${f.codecs}]")
                            // Size the decoder output buffers to the video:
                            // without this the SurfaceTexture can run small
                            // buffers and the picture upscales from mush.
                            if ((f.sampleMimeType ?: "").startsWith("video/") &&
                                f.width > 0 && f.height > 0
                            ) {
                                try {
                                    renderer.surfaceTexture?.setDefaultBufferSize(f.width, f.height)
                                    FileLog.i(TAG, "video buffer size ${f.width}x${f.height}")
                                } catch (t: Throwable) {
                                    FileLog.w(TAG, "buffer size failed: ${t.message}")
                                }
                            }
                        }
                        Log.i(TAG, sb.toString())
                        FileLog.i(TAG, sb.toString())
                    }
                })
                exo.addAnalyticsListener(object : AnalyticsListener {
                    override fun onLoadError(
                        eventTime: AnalyticsListener.EventTime,
                        loadEventInfo: LoadEventInfo,
                        mediaLoadData: MediaLoadData,
                        error: IOException,
                        wasCanceled: Boolean
                    ) {
                        // HTTP-level failures (404/416/timeouts/resets) surface here,
                        // NOT in onPlayerError — this is the money log for freezes.
                        FileLog.w(TAG, "loadError uri=${loadEventInfo.uri} " +
                            "bytes=${loadEventInfo.bytesLoaded} " +
                            "loadMs=${loadEventInfo.loadDurationMs} " +
                            "canceled=$wasCanceled err=${error.message}")
                    }
                    override fun onDroppedVideoFrames(
                        eventTime: AnalyticsListener.EventTime,
                        droppedFrames: Int,
                        elapsedMs: Long
                    ) {
                        // Climbing drops with a healthy buffer = the decoder is
                        // fine but frames never reach eyes (render-side stall).
                        FileLog.w(TAG, "droppedVideoFrames=$droppedFrames elapsedMs=$elapsedMs")
                    }
                    override fun onVideoFrameProcessingOffset(
                        eventTime: AnalyticsListener.EventTime,
                        totalProcessingOffsetUs: Long,
                        frameCount: Int
                    ) {
                        // Heartbeat: fires per batch of frames the decoder
                        // delivers to the video renderer. Timestamp it always;
                        // the watchdog reports offsetAge (growing = decoder
                        // stopped feeding, the freeze signature).
                        lastFrameBatchMs = System.currentTimeMillis()
                        // Growing offset = renderer falling behind wall clock.
                        if (frameCount > 0 && totalProcessingOffsetUs > 500_000L * frameCount) {
                            FileLog.w(TAG, "frameProcessingOffset avg=" +
                                "${totalProcessingOffsetUs / frameCount}us over $frameCount frames")
                        }
                    }
                    override fun onRenderedFirstFrame(
                        eventTime: AnalyticsListener.EventTime,
                        output: Any,
                        renderTimeMs: Long
                    ) {
                        FileLog.i(TAG, "renderedFirstFrame")
                        lastFrameBatchMs = System.currentTimeMillis()
                    }
                    override fun onVideoSizeChanged(
                        eventTime: AnalyticsListener.EventTime,
                        videoSize: androidx.media3.common.VideoSize
                    ) {
                        FileLog.i(TAG, "videoSize ${videoSize.width}x${videoSize.height}")
                    }
                    override fun onVideoEnabled(
                        eventTime: AnalyticsListener.EventTime,
                        decoderCounters: androidx.media3.exoplayer.DecoderCounters
                    ) {
                        FileLog.i(TAG, "videoEnabled")
                    }
                    override fun onVideoDisabled(
                        eventTime: AnalyticsListener.EventTime,
                        decoderCounters: androidx.media3.exoplayer.DecoderCounters
                    ) {
                        FileLog.w(TAG, "videoDisabled rendered=${decoderCounters.renderedOutputBufferCount} " +
                            "dropped=${decoderCounters.droppedBufferCount} " +
                            "skippedOut=${decoderCounters.skippedOutputBufferCount}")
                    }
                })
            }
            lifecycleScope.launch {
                for (i in 1..100) {
                    val s = renderer.surface
                    if (s != null) { withContext(Dispatchers.Main) { player?.setVideoSurface(s) }; break }
                    kotlinx.coroutines.delay(100)
                }
            }
        }
        Log.i(TAG, "play: $url")
        player?.setMediaItem(MediaItem.fromUri(url))
        player?.prepare()
        player?.playWhenReady = true
        txtStatus.text = "▶ $name  •  ${renderer.projection.label} ${renderer.stereo.label}"
    }

    private fun cycleProjection() {
        val order = listOf(Projection.FLAT, Projection.DEG180, Projection.DEG220, Projection.DEG270, Projection.DEG360, Projection.FISHEYE)
        renderer.projection = order[(order.indexOf(renderer.projection) + 1) % order.size]
        settings.projection = renderer.projection
        syncGvr()
        txtStatus.text = "${renderer.projection.label} • ${renderer.stereo.label}"
    }

    // Live rotation matrices on this stack arrive TRANSPOSED vs the AOSP
    // documented convention (proven: column-2, not row-2, equals accel-up at
    // +1.000 over 10k still samples). Transpose once on receipt so every
    // downstream consumer (basis derivation, view composition, Euler logs,
    // trace) sees docs-convention device→world matrices.
    private val trackFixed = FloatArray(16)
    private val cmpFixed = FloatArray(16)

    private val cmpListener = object : SensorEventListener {
        var lastRaw: FloatArray? = null
        override fun onSensorChanged(e: SensorEvent) {
            val raw = FloatArray(16)
            SensorManager.getRotationMatrixFromVector(raw, e.values)
            android.opengl.Matrix.transposeM(cmpFixed, 0, raw, 0)
            lastRaw = cmpFixed.clone()
        }
        override fun onAccuracyChanged(s: Sensor?, acc: Int) = Unit
    }

    /** Raw gyro/accel/mag snapshot tap for the sensor debug page. Also
     *  integrates gyro yaw about accel-up (gimbal-free turn ground truth). */
    @Volatile private var yawGyroRelDeg = 0.0
    private var lastGyroTNs = 0L
    private val envListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            when (e.sensor.type) {
                Sensor.TYPE_GYROSCOPE -> {
                    lastGyro = floatArrayOf(e.values[0], e.values[1], e.values[2])
                    // integrate yaw rate about accel-up for the TURNDET row;
                    // reject free-fall/handling spikes
                    val a = lastAccel
                    val an = kotlin.math.sqrt((a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).toDouble())
                    if (an in 7.0..13.0 && lastGyroTNs != 0L) {
                        val dt = (e.timestamp - lastGyroTNs) / 1e9
                        if (dt in 0.0..0.5) {
                            val rate = (e.values[0]*a[0] + e.values[1]*a[1] + e.values[2]*a[2]) / an
                            yawGyroRelDeg += Math.toDegrees(rate * dt)
                        }
                    }
                    lastGyroTNs = e.timestamp
                }
                Sensor.TYPE_ACCELEROMETER -> lastAccel = floatArrayOf(e.values[0], e.values[1], e.values[2])
                Sensor.TYPE_MAGNETIC_FIELD -> lastMag = floatArrayOf(e.values[0], e.values[1], e.values[2])
            }
        }
        override fun onAccuracyChanged(s: Sensor?, acc: Int) = Unit
    }

    // ---------------- head tracking ----------------
    // Pass the RAW rotation matrix. The renderer derives the screen frame
    // from gravity at recenter time, so any phone/viewer orientation works.
    override fun onSensorChanged(e: SensorEvent) {
        if (e.sensor.type != Sensor.TYPE_ROTATION_VECTOR &&
            e.sensor.type != Sensor.TYPE_GAME_ROTATION_VECTOR) return
        val v = FloatArray(5)
        System.arraycopy(e.values, 0, v, 0, minOf(e.values.size, 5))
        val raw = FloatArray(16)
        SensorManager.getRotationMatrixFromVector(raw, v)
        // See trackFixed note above: transpose to docs convention first.
        android.opengl.Matrix.transposeM(trackFixed, 0, raw, 0)
        // The renderer no longer consumes this: GVR's own head tracker
        // feeds HeadTransform each frame. Kept for the orientation
        // diagnostics, the A/B comparison and the sweep trace below.
        lastTrackM = trackFixed.clone()
        traceEvent(e.timestamp, trackFixed)
        // 1Hz orientation diagnostic for tracking issues (logcat -s LimpetVR-ori).
        // All matrices here are post-transpose (docs convention), so Euler
        // angles and up-rows are meaningful (no gimbal games beyond normal).
        if (oriLogCountdown-- <= 0) {
            oriLogCountdown = 50
            val o = FloatArray(3)
            SensorManager.getOrientation(trackFixed, o)
            val cmp = cmpListener.lastRaw
            var cmpStr = "none"
            if (cmp != null) {
                val c = FloatArray(3)
                SensorManager.getOrientation(cmp, c)
                // structural agreement: angle between the two "up" rows
                val dot = (trackFixed[2]*cmp[2] + trackFixed[6]*cmp[6] + trackFixed[10]*cmp[10])
                cmpStr = "yaw=${Math.toDegrees(c[0].toDouble()).toInt()} " +
                    "pitch=${Math.toDegrees(c[1].toDouble()).toInt()} " +
                    "roll=${Math.toDegrees(c[2].toDouble()).toInt()} " +
                    "upAgree=${String.format("%.3f", dot)}"
            }
            Log.i("LimpetVR-ori", "src=${if (useGameRv) "game" else "full"} " +
                "yaw=${Math.toDegrees(o[0].toDouble()).toInt()} " +
                "pitch=${Math.toDegrees(o[1].toDouble()).toInt()} " +
                "roll=${Math.toDegrees(o[2].toDouble()).toInt()} " +
                "upxy=(${String.format("%.2f", trackFixed[2])},${String.format("%.2f", trackFixed[6])}) " +
                "vsOther=[$cmpStr]")
        }
    }

    override fun onAccuracyChanged(s: Sensor?, acc: Int) = Unit

    private fun humanSize(n: Long): String {
        if (n < 1024) return "$n B"
        val kb = n / 1024.0
        if (kb < 1024) return String.format("%.0f KB", kb)
        val mb = kb / 1024.0
        if (mb < 1024) return String.format("%.1f MB", mb)
        return String.format("%.2f GB", mb / 1024.0)
    }
}
