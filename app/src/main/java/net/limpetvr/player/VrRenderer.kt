/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import java.util.Locale
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLUtils
import android.opengl.Matrix
import com.google.vr.sdk.base.Eye
import com.google.vr.sdk.base.GvrView
import com.google.vr.sdk.base.HeadTransform
import com.google.vr.sdk.base.Viewport
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig

/**
 * GVR/Cardboard stereo renderer (see A-method-to-render-VR-videos…md).
 *
 * VIDEO: ExoPlayer frame (OES texture) on plane / dome sphere, drawn with
 * the per-eye projection from Eye.getPerspective() and the eye view
 * (head tracking + IPD) from Eye.getEyeView(); lens distortion is the
 * Cardboard pass inside the SDK.
 * BROWSER: Canvas-rendered file/settings list on a world-locked panel.
 * Play menu: world-locked button panel (doc §7) with the app's own
 * leaky-integrator dwell, head-locked reticle / tooltip / toast (§8).
 *
 * Gaze is a real head-forward ray intersected with the panel plane, so the
 * head-locked reticle actually hovers rows. Staring [dwellMs] activates.
 * All optics (FOV scale, IPD, swap, zoom, panel distance) are live fields
 * fed from SettingsStore by the activity.
 */
class VrRenderer(
    private val onBrowserActivate: (Int, Float?) -> Unit,
    private val onMenuEvent: (MenuEvent) -> Unit = {},
    private val onWebEvent: (WebEvent) -> Unit = {}
) : GvrView.StereoRenderer, SurfaceTexture.OnFrameAvailableListener {

    enum class Mode { BROWSER, VIDEO, WEB }

    /** Web-mode events out to the activity (all marshalled to the UI thread
     *  there). HitTest is a question (answer with [onWebHit]), the rest are
     *  commands. */
    sealed class WebEvent {
        /** Ask the page what lives at page pixel (px, py). */
        data class HitTest(val token: Int, val px: Float, val py: Float) : WebEvent()
        /** Dwell finished on a link/control: activate it. */
        data class Click(val px: Float, val py: Float) : WebEvent()
        /** Scrollbar arrow dwell: -1 = page up, +1 = page down. */
        data class Scroll(val dir: Int) : WebEvent()
        /** Dwell on inert page: open (true) or close (false) the web panel. */
        data class Panel(val open: Boolean) : WebEvent()
        /** PgUp/PgDn button on the web panel: -1 up, +1 down. */
        data class PageKey(val dir: Int) : WebEvent()
    }

    /** The GvrView we render into, set by the activity. Only used for
     *  recentering (GL thread reads, activity may swap it any time). */
    @Volatile var gvrView: GvrView? = null

    /** Play-menu actions. Press(idx) index map (menuButtons ids):
     *  0 settings, 1 shape, 2 files, 3 prev, 4 rew, 5 play, 6 ff, 7 next,
     *  8 zoom+, 9 zoom-, 10 vol+, 11 vol-, 12 flip, 13 recenter,
     *  14 fov-, 15 fov+. 16 (A/V sync) and 17 (screenpos) have no button
     *  any more, so Press(16)/Press(17) never fire.
     *  Seek(frac): jump to fraction. */
    sealed class MenuEvent {
        data class Press(val idx: Int) : MenuEvent()
        data class Seek(val frac: Float) : MenuEvent()
    }

    @Volatile var mode: Mode = Mode.BROWSER
    @Volatile var projection: Projection = Projection.DEG180
    @Volatile var stereo: Stereo = Stereo.SBS
    /** Scales the Cardboard eye field of view (0.5..1.5, 1 = profile FOV). */
    @Volatile var fovScale: Float = 1f
    /** Flat-screen size multiplier (0.5..5, 1 = default). */
    @Volatile var screenSize: Float = 1f
    /** Flat-screen curvature (0 = plane, 1 = full cap bent around the
     *  viewer). Lives only in the FLAT projection; 0 keeps the old path. */
    @Volatile var screenCurve: Float = 0f
    /** Allow Cardboard lens distortion in the SDK (off = raw stereo). */
    @Volatile var disableDist: Boolean = false
    /** Show the gaze tooltip pill above the reticle. */
    @Volatile var enableTooltip: Boolean = true
    /** Dome mesh density: "vertex" (60x48) or "vertexhq" (72x60). */
    @Volatile var panoQuality: String = "vertex"
    @Volatile var swapEyes: Boolean = false
    @Volatile var zoom: Float = 1f
    @Volatile var panelDistM: Float = 2.4f
    @Volatile var dwellMs: Long = 1500L
    /** Viewer IPD in metres (GVR viewer params, default 64 mm). */
    @Volatile var ipdM: Float = 0.064f
    /** Pin video dead-ahead (screen lock); browser always look-around. */
    @Volatile var pinVideo: Boolean = false
    /**
     * Diagnostics: ignore the sensors and drive tracking with a scripted
     * sweep (yaw ±35°/10s + pitch ±12°/7s). If the image pans level in sweep
     * mode but rotates on your real head, the sensors (not the math) lie.
     */
    @Volatile var testSweep: Boolean = false
    private var lastTestSweep = false
    private var sweepT0 = 0L

    var videoTextureId: Int = -1
    /** Bumped every onSurfaceCreated. If ExoPlayer is still targeting an
     *  older surface (EGL context loss recreates it silently), its frames
     *  go nowhere: frozen picture, advancing position, zero errors. The
     *  activity watches this generation and re-attaches on change. */
    @Volatile var surfaceGen = 0
    /** Frames completed (GL thread). Watchdog reads it: advancing = GL
     *  alive; frozen + frozen video = GL stuck (GPU hang/surface stall). */
    @Volatile var frameCount = 0L
        private set
    /** Video frames actually consumed from the decoder (updateTexImage ran).
     *  The discriminator: glfps high + consumed frozen = decoder stopped
     *  delivering (input starvation/track end); both frozen = GL stalled. */
    @Volatile var consumedFrames = 0L
        private set
    /** onFrameAvailable firings (binder thread). arrivals frozen + renderer
     *  counters climbing = queue/listener stopped delivering despite output;
     *  arrivals flowing + consumed frozen = consumption broken. */
    @Volatile var arrivedFrames = 0L
        private set
    var surfaceTexture: SurfaceTexture? = null
        private set
    var surface: android.view.Surface? = null
        private set
    // Written by the BufferQueue binder thread (onFrameAvailable), read by the
    // GL thread every frame. MUST be volatile: without it the GL thread may
    // stop seeing new frames after JIT recompiles the read (seconds in) —
    // frozen video, healthy audio/position/buffers, zero errors. This exact
    // failure froze every video at varying 5-15s until found.
    @Volatile private var frameAvailable = false

    // ---------------- web (WebView -> SurfaceTexture -> OES texture) ----------------
    // The page is hosted by a WebView inside a TextureView in the activity;
    // its SurfaceTexture becomes an external GL texture sampled by the SAME
    // flat-screen path the video uses (same size, curve and zoom), so WEB
    // borrows drawVideo() wholesale rather than growing a second pipeline.
    /** Web page size in pixels (the WebView's viewport). */
    @Volatile var webPageW: Int = 1280
    @Volatile var webPageH: Int = 720
    /** Page frames uploaded (GL thread). 0 = nothing has landed: the page
     *  never rendered, or the copy failed. */
    @Volatile var webConsumedFrames: Long = 0L
        private set
    /** Hand a freshly captured page bitmap to the GL thread (UI thread).
     *  The bitmap is reused by the caller, so it is uploaded on the very
     *  next frame — the copy must not be mutated before then. */
    fun submitWebFrame(bmp: Bitmap) { webBmpPending = bmp }
    /** Answer to a [WebEvent.HitTest]: is there something clickable at the
     *  page pixel that token asked about? */
    fun onWebHit(token: Int, interactive: Boolean) {
        if (token == webHitToken) webHitInteractive = interactive
    }
    /** Page scroll state from the activity (thumb position + scrollable). */
    @Volatile var webScrollFrac: Float = 0f
    /** Visible fraction of the page (0..1) — sets the thumb's size. */
    @Volatile var webViewFrac: Float = 1f
    @Volatile var webScrollable: Boolean = false
    /** The web control panel is open (gaze drives the panel, not the page). */
    @Volatile var webPanelOpen: Boolean = false

    @Volatile private var webBmpPending: Bitmap? = null
    private var webTexId: Int = -1
    private var webBarTexId: Int = -1
    private var webBarBitmap: Bitmap? = null
    private var webBarMesh: Mesh? = null
    private var webBarMeshKey = ""
    @Volatile private var webBmpConsumed = 0L
    private var webHitToken = 0
    @Volatile private var webHitInteractive = false
    private var webHitAskedAt = 0L
    private var webAskPx = -1f
    private var webAskPy = -1f
    /** Page pixels of head wobble still considered "the same place", so a
     *  slightly stale hit answer is trusted rather than flickering. */
    private val WEB_HIT_SLOP = 70f
    private var webLastClickAt = 0L
    // dwell state for the page itself
    private var webProgF = 0f
    private var webProgT = 0L
    private var webTarget = ""      // "" = inert page, else "px,py"
    private var webFiredFor = "\u0000none"
    private var webPanelHold = 0f
    private var webPanelT0 = 0L
    private var webPanelLockUntil = 0L
    /** TEMP DEBUG: where the gaze samples, for the on-page crosshair. */
    @Volatile var webDebugPoint: FloatArray? = null
    private var webDbgT0 = 0L
    @Volatile var webDbgOn: Boolean = false // TEMP DEBUG
    private var webScrollDir = 0
    private var webScrollFiredFor = 0
    private val webStill = MotionStillness(400L, 5f)

    @Volatile var browserTitle: String = "/"
    @Volatile var browserRows: List<BrowserRow> = emptyList()
    /** A row can carry a gaze slider: dwelling at horizontal fraction u
     *  sets value = min + u·(max-min). One dwell reaches any value. */
    /** Display format for a gaze slider's live tooltip: display value =
     *  raw * scale + offset, snapped to the snap grid (0 = no snap),
     *  rendered with decimals places plus suffix. Mirrors handleSlide. */
    data class SlideFormat(
        val suffix: String = "",
        val decimals: Int = 0,
        val scale: Float = 1f,
        val offset: Float = 0f,
        val snap: Float = 0f
    )
    data class BrowserRow(
        val label: String, val meta: String, val kind: Int,
        val slideKey: String? = null,
        val slideMin: Float = 0f,
        val slideMax: Float = 1f,
        val slideVal: Float = 0f,
        val slideFmt: SlideFormat? = null,
        val segLabels: List<String> = emptyList(),
        val segActions: List<String> = emptyList(),
        val segSelected: Int = -1,
        val previewMags: FloatArray? = null, // shaping preview: per-point 0..1
        val previewHull: IntArray? = null, // shaping preview: convex-hull indices
        val previewN: Int = 9, // shaping preview grid size
        val previewPos: FloatArray? = null, // shaping preview: absolute [x,y] per point, [0,1], y down
        val dead: Boolean = false // rest zone: hover drains, never accumulates or fires
    ) {
        companion object {
            const val FOLDER = 0; const val VIDEO = 1; const val FILE = 2; const val ACTION = 3
        }
    }

    // highlight index into FULL rows list
    private var highlight = -1
    /** Top edge of the rows window, in row units. File pages glide it
     *  fractionally while a scroll strip is engaged; settings pages snap
     *  it to integers via ensureVisible. Counts from the first SCROLLING
     *  row (after the pinned top rows). */
    private var scrollPos = 0f
    /** Pinned top rows: file pages pin home + up = 2 (scroll strips on),
     *  settings/shaping/sensor pages 0 (plain window, no strips). Set by
     *  the activity per page in pushRows. */
    @Volatile var pinTopRows = 0
    /** PgUp/PgDn buttons, right side of the title bar. Web panel only: the
     *  file and SMB lists have no page keys to send. */
    @Volatile var webSideBtns = false
    private var sideBtnDir = 0
    private var sideBtnProg = 0f
    private var sideBtnFired = false
    /** Scroll-strip state (GL thread): 0 idle, -1 scrolling up, +1 down. */
    private var scrollEngage = 0
    /** Strip currently earning trigger progress (same sign convention). */
    private var scrollTrigDir = 0
    /** Trigger progress 0..1: short still-gaze on a strip engages it. */
    private var scrollTrigF = 0f
    private var dwellStart = 0L
    private var dwellFiredFor = -2
    // X close button (title bar, top right): own dwell state, fires sentinel -10.
    // Hit zone in panel TEX coords: x > 920, y < TITLE_Y1 (title bar).
    private var inXZone = false
    private var xProgF = 0f
    private var xDwellFired = false
    // Diagnostic: log once per latch episode when a completed dwell is
    // suppressed by the fired latch (tells stuck-latch from no-dwell).
    private var fireBlockedLogged = false
    fun tapSelect() { val h = highlight; if (h in browserRows.indices) onBrowserActivate(h, null) }
    fun moveHighlight(d: Int) {
        val n = browserRows.size
        if (n == 0) return
        highlight = ((if (highlight < 0) n / 2 else highlight) + d).coerceIn(0, n - 1)
        ensureVisible()
        dwellStart = now(); dwellFiredFor = -2
    }

    // Head pose from GVR (doc §3): headView = world→head (HeadTransform),
    // headWorld = headView · N where N is the constant §3 basis shift that
    // puts the viewer 0.01 m in front of the world origin. invHeadWorld
    // turns the head-forward ray into a world ray for gaze. The monitor is
    // headViewM: stillness gates and copies lock on it.
    private val headViewM = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
    private val headWorldM = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
    private val invHeadWorldM = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
    /** N (doc §3): translate the viewer 0.01 m along -Z of world space.
     *  Frozen — the live basis below is re-folded against it on recenter. */
    private val basisShiftN = FloatArray(16).also {
        Matrix.setLookAtM(it, 0, 0f, 0f, 0.01f, 0f, 0f, 0f, 0f, 1f, 0f)
    }
    /** appWorld→trackerWorld, initially just N. Recenter overwrites it with
     *  headView⁻¹·N so the CURRENT head pose — yaw, pitch AND roll — becomes
     *  the app world (see the recenterPending block in frameTick). */
    private val basisShiftM = FloatArray(16).also {
        System.arraycopy(basisShiftN, 0, it, 0, 16)
    }
    private val tmpA = FloatArray(16)
    private val tmpB = FloatArray(16)
    /** Rendered camera-forward, refreshed every frame (for the trace recorder). */
    @Volatile var lastEffFwd = floatArrayOf(0f, 0f, -1f)
    /** Rendered head-up, refreshed every frame. Drives the menu trigger. */
    @Volatile var lastEffUp = floatArrayOf(0f, 1f, 0f)
    /** Cause tag for the next snap (entry/files/video/tap/auto). Audit trail. */
    @Volatile private var snapTag = "auto"
    /** Successful basis snaps since creation (debug overlay). */
    @Volatile var snapCount = 0
        private set
    /** Kept frames: no longer derivable with GVR tracking (stays 0). */
    @Volatile var keptFrames = 0
        private set

    /** Head orientation in world space (debug overlay / sensor page): the
     *  head→world basis (N baked in), so forward/up read off it directly. */
    fun effCopy(): FloatArray = synchronized(headViewM) { invHeadWorldM.clone() }

    /** Consume the request on the GL thread (GVR recenter is a native
     *  call, safe there) so taps and the aim dwell snap immediately. */
    @Volatile private var recenterPending = false

    /** Snap the world to the current head pose — full 3DOF (yaw, pitch and
     *  roll), consumed on the GL thread in frameTick. */
    fun recenter(why: String = "auto"): Boolean {
        if (mode == Mode.WEB) webPanelLockUntil = now() + (PANEL_TOGGLE_MS * 4).toLong()
        android.util.Log.d("LimpetVR-basis", "recenter ($why)")
        try { FileLog.d("LimpetVR-basis", "recenter ($why)") } catch (_: Throwable) {}
        snapTag = "auto"
        recenterPending = true
        inputGraceUntil = now() + 800
        return true
    }

    /** Re-center on the next frame (enter VR / play). */
    fun resetBasis(tag: String = "auto") {
        snapTag = tag
        recenterPending = true
        inputGraceUntil = now() + 2500
    }

    @Volatile private var inputGraceUntil = 0L

    private var progOes = 0; private var prog2d = 0; private var progWeb = 0
    private var aPosOes = 0; private var aTexOes = 0; private var uMvpOes = 0
    private var uTexOes = 0; private var uStereoOes = 0; private var uEyeOes = 0
    private var uTexMatOes = 0; private var uZoomOutOes = 0
    // Fisheye circle-sampling path (§8): same vertex shader, dedicated frag.
    private var progFish = 0
    private var aPosFish = 0; private var aTexFish = 0; private var uMvpFish = 0
    private var uTexFish = 0; private var uStereoFish = 0; private var uEyeFish = 0
    private var uTexMatFish = 0; private var uZoomOutFish = 0
    private var uFishC = 0; private var uFishR = 0; private var uFishMirror = 0
    private var uWarpOnOes = 0; private var uWarpCxOes = 0; private var uWarpK1Oes = 0; private var uWarpK2Oes = 0; private var uWarpAspectOes = 0
    private var aPos2d = 0; private var aTex2d = 0; private var uMvp2d = 0; private var uTex2d = 0
    private var uAlpha2d = 0
    private var aPosWeb = 0; private var aTexWeb = 0
    private var uMvpWeb = 0; private var uTexWeb = 0; private var uZoomWeb = 0

    private var mesh: Mesh? = null
    private var meshKey: String = ""
    private var browserTexId = -1
    private var browserBitmap: Bitmap? = null
    private var lastPanelHash = 0
    private var reticleTexId = -1
    /** Blue twin of the dwell reticle, for the recenter aim pointer. */
    private var aimTexId = -1

    /** Per-eye projections from Eye.getPerspective() plus the convergence
     *  trim, and the per-eye view O = eye.getEyeView() · N (doc §3). ov is
     *  the MVP base for every panel, the reticle and the FLAT screen;
     *  domeOv is that same view under projDomeM — the FOV-scaled panoramic
     *  projection — and feeds the dome/fisheye video only (§6.3). */
    private val projM = FloatArray(16)
    private val eyeViewNM = FloatArray(16)
    private val ovM = FloatArray(16)
    private val ptrTmp3 = FloatArray(3)
    private val projDomeM = FloatArray(16)
    private val domeOvM = FloatArray(16)
    /** Eye translation in head space (eyeView · headView⁻¹), pinVideo only. */
    private val eyeShiftM = FloatArray(16)
    private val modelM = FloatArray(16)
    /** Convergence trim: uniform clip-space offset per eye. An m[8]
     *  addition shifts the image by minus that amount in NDC x, so the
     *  left eye takes -ct: positive trim converges the halves (§10).
     *  Baseline -0.040; live-tunable after. */
    @Volatile var convTrimNdc = -0.04f
    /** Play-menu trigger tilts: up opens the top menu, down the bottom menu. */
    @Volatile var menuAngleUp = 40f
    @Volatile var menuAngleDown = -40f
    /** Which side the play menu lives on; flipped by its ⇅ button (persisted). */
    @Volatile var menuSideUp = true
    @Volatile private var menuAnimFrom = 52f
    @Volatile private var menuAnimT0 = 0L
    private val menuAnimMs = 350L
    /** Flip top<->bottom with a quick visible sweep through the middle. */
    fun menuToggleSide() {
        menuAnimFrom = menuElevCurrent()
        menuSideUp = !menuSideUp
        menuAnimT0 = now()
    }
    /** Browser panel elevation (deg): 0 = centered at horizon (file
     *  browsing), halfway to the play menu when floating over video. */
    @Volatile var browserElevDeg: Float = 0f
    /** Elevation for browser panels floating over live video: halfway
     *  between center (horizon) and the play-menu panel. */
    fun overlayElevDeg(): Float = menuElevDeg() / 2f
    /** True while the play menu is shown (VIDEO mode only). */
    @Volatile var menuOpen = false
    /** Transient in-VR message on the menu panel (Toasts are invisible
     *  in the headset). Activity sets text; visible ~2s. */
    @Volatile var menuFlash = ""
    @Volatile var menuFlashUntil = 0L
    fun flashMenu(msg: String, ms: Long = 2000L) {
        menuFlash = msg
        menuFlashUntil = System.currentTimeMillis() + ms
        showToast(msg, ms)
    }
    /** Center-screen 3D toast (Android Toasts are unreadable in-headset):
     *  head-locked quad in the vertical middle, auto-expiring. */
    @Volatile var toastText = ""
    @Volatile var toastUntil = 0L
    fun showToast(msg: String, ms: Long = 2500L) {
        toastText = msg
        toastUntil = System.currentTimeMillis() + ms
    }
    private var toastTexId = 0
    private var tipTexId = 0
    private var lastToastText = ""
    private var lastToastActive = false
    private fun maybeUploadToast() {
        val active = toastText.isNotEmpty() && now() < toastUntil
        if (active == lastToastActive && toastText == lastToastText && toastBitmap != null) return
        lastToastActive = active
        lastToastText = toastText
        val W = 512; val H = 112
        val bmp = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.TRANSPARENT)
        if (active) {
            val p = Paint(Paint.ANTI_ALIAS_FLAG)
            p.color = Color.argb(220, 10, 14, 22)
            c.drawRoundRect(4f, 4f, (W - 4).toFloat(), (H - 4).toFloat(), 24f, 24f, p)
            p.color = Color.WHITE; p.textSize = 40f; p.textAlign = Paint.Align.CENTER
            c.drawText(toastText.take(34), W / 2f, 70f, p)
            p.textAlign = Paint.Align.LEFT
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, toastTexId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        toastBitmap?.recycle()
        toastBitmap = bmp
    }
    private var toastBitmap: Bitmap? = null
    /** Toast pill, head-locked in 3D so it is a STEREO pair: the same
     *  head-space quad path as the reticle/tooltip (ov through invHeadWorld),
     *  centred on the gaze ray at panel depth. The old path drew an NDC quad
     *  through the identity matrix — the identical image in both eye
     *  viewports, zero disparity, which fuses into ONE flat picture glued to
     *  the screen. */
    private fun drawToast() {
        maybeUploadToast()
        if (toastText.isEmpty() || now() >= toastUntil) return
        val k = menuScale()
        putQuad(ptrVerts, ptrTex, -TOAST_HALF_W * k, -TOAST_HALF_H * k, TOAST_HALF_W * k, TOAST_HALF_H * k)
        headLockedAt(tmpA, 0f, 0f, -panelDistM)
        Matrix.multiplyMM(mvpM, 0, ovM, 0, tmpB, 0)
        drawQuadTex(toastTexId, mvpM)
    }

    /** Playback position/duration/state for the menu progress bar. */
    @Volatile var menuPosMs = 0L
    @Volatile var menuDurMs = 0L
    @Volatile var menuPlaying = true
    /** File name shown across the top of the play menu. */
    @Volatile var menuTitle = ""
    /** Rewind/fast-forward jump, seconds (2D setting). */
    @Volatile var skipSecs = 10
    // menu gaze state (GL thread)
    private var menuHighlight = -2 // -1 = seek bar, 0..17 buttons, -2 = decorative
    private var menuDwellFiredFor = -3
    private var menuHitValid = false
    // hovered seek fraction (panel design x) for the live time tooltip; -1 = none
    private var menuSeekHoverU = -1f
    /** Last ray/plane hit in design panel space (menuHitTest). */
    private var menuHitU = 0f
    private var menuHitV = 0f
    private var menuHitId = -2
    /** Head-locked gaze tooltip pill (updated by updateTooltip). */
    private var tooltipVisible = false
    private var tooltipText = ""
    private var lastTipText: String? = null
    private var tipBitmap: Bitmap? = null
    private var menuTexId = 0
    private var menuBitmap: Bitmap? = null
    private var lastMenuHash = 0
    /** When the gaze first dropped below the close threshold (0 = above).
     *  Closing needs 400ms continuously below: sensor noise and transient
     *  tilt dips while operating the end buttons must not strobe the menu
     *  (and reset every dwell). */
    private var menuBelowSince = 0L
    private var menuWasOpen = false
    private var menuProgFresh = true
    /** Cardboard lens distortion coefficients (0..1, standard Cardboard).
     *  The activity pushes them into the SDK's Distortion object; the
     *  actual lens pass runs inside the GVR native renderer. */
    @Volatile var lensK1 = 0.34f
    @Volatile var lensK2 = 0.55f
    /** Master strength for the lens pass (1 = physical, 0 = off). */
    @Volatile var lensStrength = 1f
    private val mvpM = FloatArray(16)
    private val tmpM = FloatArray(16)

    /** Decoded-frame aspect (width/height) for fisheye circle calibration.
     *  Refreshed by the activity from the video track when known. */
    @Volatile var videoAspect = 16f / 9f
    /** Fisheye circle calibration (§8): radius multiplier (1 = default),
     *  center offsets in frame UV, per-eye horizontal-mirror flags. */
    @Volatile var fisheyeRadiusScale = 1f
    @Volatile var fisheyeCxOff = 0f
    @Volatile var fisheyeCyOff = 0f
    @Volatile var fisheyeMirrorL = false
    @Volatile var fisheyeMirrorR = false
    /** Decoder texture transform (SurfaceTexture.getTransformMatrix),
     *  refreshed every frame on the GL thread and honored by both video
     *  sampling paths (§4). Identity until the first frame arrives. */
    private val texMat = FloatArray(16).also { Matrix.setIdentityM(it, 0) }

    @Volatile var lastWidth = 1
    @Volatile var lastHeight = 1

    companion object {
        /** Play-menu panel world geometry (doc §7): a button is 0.6 m
         *  across at pitch 0.75 m on a panel of radius panelDistM. */
        const val MENU_BTN_HALF = 0.3f
        const val MENU_PITCH = 0.75f
        /** Panel texture covers x ∈ [MENU_X0, MENU_X1], y ∈ [MENU_Y0, MENU_Y1]
         *  in panel space (y up), 1024 texels wide. */
        const val MENU_X0 = -5.3f
        const val MENU_X1 = 5.3f
        const val MENU_Y0 = -2.0f
        const val MENU_Y1 = 1.4f
        const val MENU_TEX_W = 1024
        const val MENU_TEX_H = 328
        /** Viewing distance the panel rect was authored for (the flat screen
         *  sits at −11.95 too); the live rect scales with panelDistM. */
        const val MENU_DESIGN_R = 11.95f
        /** Viewing distance of the flat screen plane (and the cap's blend
         *  start): buildVideoModel's z translate, screenCapMesh's flat leg. */
        const val FLAT_DIST = 11.95f
        /** Tightest cap radius at curve = 1: the picture bends toward the
         *  viewer but never closer than this (metres). */
        const val SCREEN_R_MIN = 4.0f
        /** Wrap clamp: the bent screen never sweeps past this total angle,
         *  so its edges stay in front of the ears (not a whole sphere). */
        const val SCREEN_ARC_MAX_DEG = 150f
        /** Rect half-extents: x ±5.3, y −2.0..1.4 (centre −0.3, half 1.7). */
        const val MENU_RECT_HW = 5.3f
        const val MENU_RECT_HH = 1.7f
        const val MENU_RECT_VOFF = -0.3f
        /** Near-edge extent past the panel centre, per side (bottom 1.8,
         *  top 1.05 — title top) for the elevation formula. */
        const val MENU_EXT_BOT = 1.8f
        const val MENU_EXT_TOP = 1.05f
        /** Degrees the near edge keeps beyond the trigger angle. */
        const val MENU_MARGIN_DEG = 6.7f
        /** Reticle diameter angle (rad): world half-size = sin(0.03)·R. */
        private const val RETICLE_ANG = 0.03f
        /** Gaze tooltip pill, half-extents at panel depth (design units). */
        const val TIP_HALF_W = 1.28f
        const val TIP_HALF_H = 0.24f
        /** Toast pill at panel depth (design units), aspect-matched to the
         *  512×112 texture; ~62%×14% of the view, like the old NDC quad. */
        const val TOAST_HALF_W = 4.3f
        const val TOAST_HALF_H = 0.94f
        /** Screen-position calibration sweep, milliseconds. */
        const val SCREEN_SWEEP_MS = 4000f
        const val TEX = 1024
        const val ROW_H = 64
        // ---- browser panel layout (TEX coords, y down) ----
        // Settings pages: title bar, then a plain 13-row window.
        // File pages (pin 2): title bar, pinned home + up rows, scroll-up
        // strip, scrolling window, scroll-down strip. The strips are pinned
        // chrome: always visible whenever the list overflows the window.
        // The window holds 12 rows total shared between pinned and
        // scrolling rows, so the down strip always lands at the same y.
        const val TITLE_Y1 = 110
        const val ROWS_Y0 = 150
        const val VISIBLE_ROWS = 13
        const val PIN_Y0 = 114
        const val STRIP_H = 56
        // PgUp/PgDn buttons, top-right of the panel (inside the title bar,
        // which the short web title leaves free).
        const val SIDE_BTN_X0 = 856f
        const val SIDE_BTN_X1 = 1004f
        const val SIDE_BTN_H = 46f
        const val SIDE_BTN_UP_Y0 = 12f
        const val SIDE_BTN_DN_Y0 = 62f
        /** Rows visible in the scrolling window (12 minus pinned rows). */
        fun winRows(pin: Int): Int = 12 - pin.coerceIn(0, 2)
        /** Top y of the scroll-up strip. */
        fun upStripY0(pin: Int): Float = (PIN_Y0 + pin.coerceIn(0, 2) * ROW_H).toFloat()
        /** Top y of the scrolling rows window. */
        fun rowsY0(pin: Int): Float = upStripY0(pin) + STRIP_H
        /** Top y of the scroll-down strip (same for pin 0..2 by design). */
        fun downStripY0(pin: Int): Float = rowsY0(pin) + winRows(pin) * ROW_H
        // Scroll-strip feel: still-gaze time to engage, then rows/second.
        const val STRIP_TRIG_MS = 350f
        const val STRIP_ROWS_PER_SEC = 10f
        // Panel open/close fade: fast smoothstep, both directions.
        const val PANEL_FADE_MS = 180f
    /** How long the gaze must rest on the web panel to toggle it. */
    const val PANEL_TOGGLE_MS = 350f

        private const val VERT = """
attribute vec4 aPos; attribute vec2 aTex; varying vec2 vTex; varying vec3 vDir; uniform mat4 uMvp;
uniform float uWarpOn; uniform float uWarpCx; uniform float uWarpK1; uniform float uWarpK2; uniform float uWarpAspect;
void main(){
  vTex = aTex;
  vDir = aPos.xyz;
  vec4 p = uMvp * aPos;
  float w = p.w;
  if (uWarpOn > 0.5) {
    if (w > 0.0) {
      float nx = p.x / w;
      float ny = p.y / w;
      float dx = (nx - uWarpCx) * uWarpAspect;
      float r2 = dx * dx + ny * ny;
      float s = 1.0 / (1.0 + uWarpK1 * r2 + uWarpK2 * r2 * r2);
      nx = uWarpCx + dx * s / uWarpAspect;
      ny = ny * s;
      p.x = nx * w;
      p.y = ny * w;
    }
  }
  gl_Position = p;
}
"""
        // highp UVs when available: mediump quantizes texture coordinates
        // to ~1024 steps, i.e. 8-texel blocks on 8K video ("lego").
        // Equirectangular dome sampling (§2, §4): the mesh vertex UV already
        // maps yaw/pitch linearly across the span; the stereo half is
        // selected here and the decoder transform honored. Zoom-in is the
        // dome model sliding toward the viewer (§5) — never the frustum;
        // only zoom-out touches UVs, minifying about the half-image center
        // so the sampled range never widens.
        private const val FRAG_OES = """
#extension GL_OES_EGL_image_external : require
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 vTex;
varying vec3 vDir;
uniform samplerExternalOES uTex; uniform int uStereo; uniform int uEye;
uniform mat4 uTexMat; uniform float uZoomOut;
void main(){
  vec2 t = vTex;
  if (uStereo == 1) { t.x = (t.x + float(uEye)) * 0.5; }
  else if (uStereo == 2) { t.y = (t.y + float(uEye)) * 0.5; }
  // Zoom-out minifies in texture space about the half-image center (§5).
  // uZoomOut is 1 for zoom >= 1 (the dome model / FLAT window owns that
  // range); below
  // 1 the sampled range EXPANDS (divide), so the picture shrinks toward
  // the half center instead of widening the frustum into the rim zone.
  // The half-bounds check below clamps the uncovered margin to black, so
  // one eye never bleeds into the other's.
  vec2 c = vec2(0.5);
  if (uStereo == 1) { c = vec2(float(uEye) * 0.5 + 0.25, 0.5); }
  else if (uStereo == 2) { c = vec2(0.5, float(uEye) * 0.5 + 0.25); }
  t = c + (t - c) / uZoomOut;
  // Bounds check per half-image (not full [0,1]): each eye only sees its
  // half. A full-range check lets zoom-out bleed the other eye's half in
  // at the extremes.
  bool oob = false;
  if (uStereo == 1) {
    float halfMin = float(uEye) * 0.5;
    float halfMax = halfMin + 0.5;
    oob = (t.x < halfMin || t.x > halfMax || t.y < 0.0 || t.y > 1.0);
  } else if (uStereo == 2) {
    float halfMin = float(uEye) * 0.5;
    float halfMax = halfMin + 0.5;
    oob = (t.y < halfMin || t.y > halfMax || t.x < 0.0 || t.x > 1.0);
  } else {
    oob = (t.x < 0.0 || t.x > 1.0 || t.y < 0.0 || t.y > 1.0);
  }
  if (oob) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  vec4 st = uTexMat * vec4(t, 0.0, 1.0);
  gl_FragColor = texture2D(uTex, st.xy);
}
"""
        // Fisheye circle sampling (§8): each pixel's view direction is
        // recovered from the interpolated mesh position (the dome is
        // centered on the origin, so normalize(aPos) is the view ray).
        // Angle from the forward axis (-Z) over a hemisphere gives the
        // equidistant radius; the bearing gives the direction in the
        // circle. Outside the active circle (corners included) is black.
        // Mesh, basis, projection, zoom and lens pass are identical to the
        // equirectangular path.
        private const val FRAG_OES_FISH = """
#extension GL_OES_EGL_image_external : require
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 vTex;
varying vec3 vDir;
uniform samplerExternalOES uTex; uniform int uStereo; uniform int uEye;
uniform mat4 uTexMat; uniform float uZoomOut;
uniform vec2 uFishC; uniform vec2 uFishR; uniform float uFishMirror;
void main(){
  vec3 d = normalize(vDir);
  float cosA = clamp(-d.z, -1.0, 1.0);
  float ang = acos(cosA);
  float maxAng = 1.5707963;
  if (ang > maxAng) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  float s = sin(ang);
  vec2 bearing = (s > 1e-4) ? (d.xy / s) : vec2(0.0, 0.0);
  float rr = ang / maxAng;
  vec2 off = bearing * rr;
  if (uFishMirror > 0.5) { off.x = -off.x; }
  if (dot(off, off) > 1.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  vec2 t = uFishC + vec2(off.x * uFishR.x, off.y * uFishR.y);
  // Zoom-out minifies in texture space about the half-image center (§5),
  // same as the equirect path: the frustum never widens.
  vec2 zc = vec2(0.5);
  if (uStereo == 1) { zc = vec2(float(uEye) * 0.5 + 0.25, 0.5); }
  else if (uStereo == 2) { zc = vec2(0.5, float(uEye) * 0.5 + 0.25); }
  t = zc + (t - zc) / uZoomOut;
  bool oob = false;
  if (uStereo == 1) {
    float halfMin = float(uEye) * 0.5;
    float halfMax = halfMin + 0.5;
    oob = (t.x < halfMin || t.x > halfMax || t.y < 0.0 || t.y > 1.0);
  } else if (uStereo == 2) {
    float halfMin = float(uEye) * 0.5;
    float halfMax = halfMin + 0.5;
    oob = (t.y < halfMin || t.y > halfMax || t.x < 0.0 || t.x > 1.0);
  } else {
    oob = (t.x < 0.0 || t.x > 1.0 || t.y < 0.0 || t.y > 1.0);
  }
  if (oob) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  vec4 st = uTexMat * vec4(t, 0.0, 1.0);
  gl_FragColor = texture2D(uTex, st.xy);
}
"""
        private const val FRAG_2D = """
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 vTex; varying vec3 vDir; uniform sampler2D uTex; uniform float uAlpha;
void main(){ gl_FragColor = texture2D(uTex, vTex) * uAlpha; }
"""
        /** Web page: the panel shader plus the flat video's centre zoom, so
         *  the browser magnifies over the same 0.1..20x range. */
        private const val FRAG_WEB = """
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 vTex; varying vec3 vDir; uniform sampler2D uTex; uniform float uZoom;
void main(){
  vec2 c = vec2(0.5);
  vec2 t = c + (vTex - c) / uZoom;
  // The flat-screen mesh is wound for the video's external textures (v=0 is
  // the BOTTOM row); a bitmap capture is the other way up, so flip it here
  // rather than disturbing the shared mesh.
  t.y = 1.0 - t.y;
  if (t.x < 0.0 || t.x > 1.0 || t.y < 0.0 || t.y > 1.0) { gl_FragColor = vec4(0.0,0.0,0.0,1.0); return; }
  gl_FragColor = texture2D(uTex, t);
}
"""
    }

    override fun onSurfaceCreated(config: EGLConfig?) {
        // Black: the warped meshes cover less than the viewport, and the
        // lens boundary must read as darkness, like real VR software.
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        // Alpha blending for the pointer ring (transparent bitmap
        // background). Opaque content (video, panels) has alpha 1, so this
        // is a no-op for everything except the pointer.
        // PREMULTIPLIED alpha: every 2-D texture is an Android bitmap (which
        // stores color already multiplied by coverage), so the shader writes
        // (src·a) and the blend must consume it as-is: ONE for src,
        // (1−src.a) for dst. Straight-alpha blending here would double-dark
        // every faded panel (browser fades, menu dwell dimming).
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glDisable(GLES20.GL_CULL_FACE)
        progOes = buildProgram(VERT, FRAG_OES)
        aPosOes = GLES20.glGetAttribLocation(progOes, "aPos")
        aTexOes = GLES20.glGetAttribLocation(progOes, "aTex")
        uMvpOes = GLES20.glGetUniformLocation(progOes, "uMvp")
        uTexOes = GLES20.glGetUniformLocation(progOes, "uTex")
        uStereoOes = GLES20.glGetUniformLocation(progOes, "uStereo")
        uEyeOes = GLES20.glGetUniformLocation(progOes, "uEye")
        uTexMatOes = GLES20.glGetUniformLocation(progOes, "uTexMat")
        uZoomOutOes = GLES20.glGetUniformLocation(progOes, "uZoomOut")
        progFish = buildProgram(VERT, FRAG_OES_FISH)
        aPosFish = GLES20.glGetAttribLocation(progFish, "aPos")
        aTexFish = GLES20.glGetAttribLocation(progFish, "aTex")
        uMvpFish = GLES20.glGetUniformLocation(progFish, "uMvp")
        uTexFish = GLES20.glGetUniformLocation(progFish, "uTex")
        uStereoFish = GLES20.glGetUniformLocation(progFish, "uStereo")
        uEyeFish = GLES20.glGetUniformLocation(progFish, "uEye")
        uTexMatFish = GLES20.glGetUniformLocation(progFish, "uTexMat")
        uZoomOutFish = GLES20.glGetUniformLocation(progFish, "uZoomOut")
        uFishC = GLES20.glGetUniformLocation(progFish, "uFishC")
        uFishR = GLES20.glGetUniformLocation(progFish, "uFishR")
        uFishMirror = GLES20.glGetUniformLocation(progFish, "uFishMirror")
        uWarpOnOes = GLES20.glGetUniformLocation(progOes, "uWarpOn")
        uWarpCxOes = GLES20.glGetUniformLocation(progOes, "uWarpCx")
        uWarpK1Oes = GLES20.glGetUniformLocation(progOes, "uWarpK1")
        uWarpK2Oes = GLES20.glGetUniformLocation(progOes, "uWarpK2")
        uWarpAspectOes = GLES20.glGetUniformLocation(progOes, "uWarpAspect")
        prog2d = buildProgram(VERT, FRAG_2D)
        aPos2d = GLES20.glGetAttribLocation(prog2d, "aPos")
        aTex2d = GLES20.glGetAttribLocation(prog2d, "aTex")
        uMvp2d = GLES20.glGetUniformLocation(prog2d, "uMvp")
        uTex2d = GLES20.glGetUniformLocation(prog2d, "uTex")
        uAlpha2d = GLES20.glGetUniformLocation(prog2d, "uAlpha")
        progWeb = buildProgram(VERT, FRAG_WEB)
        aPosWeb = GLES20.glGetAttribLocation(progWeb, "aPos")
        aTexWeb = GLES20.glGetAttribLocation(progWeb, "aTex")
        uMvpWeb = GLES20.glGetUniformLocation(progWeb, "uMvp")
        uTexWeb = GLES20.glGetUniformLocation(progWeb, "uTex")
        uZoomWeb = GLES20.glGetUniformLocation(progWeb, "uZoom")

        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        videoTextureId = tex[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        surfaceTexture = SurfaceTexture(videoTextureId)
        surfaceTexture?.setOnFrameAvailableListener(this)
        surface = android.view.Surface(surfaceTexture)
        surfaceGen++

        GLES20.glGenTextures(1, tex, 0)
        browserTexId = tex[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, browserTexId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        // Web page texture: a plain 2D texture fed by PixelCopy captures of
        // the WebView, drawn by drawWeb() on the flat screen.
        GLES20.glGenTextures(1, tex, 0)
        webTexId = tex[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, webTexId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        GLES20.glGenTextures(1, tex, 0)
        webBarTexId = tex[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, webBarTexId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        reticleTexId = makeReticle(Color.RED)
        aimTexId = makeReticle(Color.rgb(96, 165, 250))

        GLES20.glGenTextures(1, tex, 0)
        menuTexId = tex[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, menuTexId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        GLES20.glGenTextures(1, tex, 0)
        toastTexId = tex[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, toastTexId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        GLES20.glGenTextures(1, tex, 0)
        tipTexId = tex[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tipTexId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        meshKey = "" // meshes survive, but rebuild against the new session
        browserGrid = null
        menuGrid = null
        webBarMesh = null
        webBarMeshKey = ""
        lastPanelHash = 0
        lastMenuHash = 0
        lastTipText = null
    }

    override fun onSurfaceChanged(w: Int, h: Int) {
        lastWidth = w.coerceAtLeast(1)
        lastHeight = h.coerceAtLeast(1)
    }

    /** Once per frame, before either eye: consume the decoded frame, fold
     *  the head pose into the §3 matrices, run gaze/menu dwell. */
    override fun onNewFrame(head: HeadTransform) {
        try {
            frameTick(head)
            frameCount++
        } catch (e: Throwable) {
            // An uncaught exception here kills the GL thread SILENTLY:
            // frozen picture, advancing position, zero errors. Never again.
            android.util.Log.e("LimpetVR-GL", "onNewFrame failed", e)
            FileLog.e("LimpetVR-GL", "onNewFrame failed", e)
        }
    }

    override fun onDrawEye(eye: Eye) {
        try {
            drawEye(eye)
        } catch (e: Throwable) {
            android.util.Log.e("LimpetVR-GL", "onDrawEye failed", e)
            FileLog.e("LimpetVR-GL", "onDrawEye failed", e)
        }
    }

    override fun onFinishFrame(viewport: Viewport?) = Unit

    override fun onRendererShutdown() = Unit

    private var texFailCount = 0

    private fun frameTick(head: HeadTransform) {
        // Consume UNCONDITIONALLY once any frame has ever arrived (the queue
        // warms within ~1s of start; before that stay flag-guarded so an
        // empty queue never throws). Gating consumption on the flag
        // deadlocks permanently: if the producer ever gets a frame ahead
        // (ordinary jitter), its overflow replaces the queued frame SILENTLY
        // (no onFrameAvailable), the flag stays false forever, we never
        // consume, the queue never drains — frozen video, healthy audio,
        // zero errors, both decoders, varying 5-25s. Latching the same frame
        // an extra time is harmless; a stale slot is fatal. Never again.
        // NOTE: no mode check here — panels float over LIVE video in
        // BROWSER mode now, so the queue must keep draining there too.
        surfaceTexture?.let { st ->
            if (frameAvailable || arrivedFrames > 0) {
                try { st.updateTexImage(); if (frameAvailable) consumedFrames++ } catch (e: Throwable) { texFailCount++; if (texFailCount <= 3 || texFailCount % 300 == 0) { android.util.Log.e("LimpetVR-GL", "updateTexImage failed #$texFailCount", e); FileLog.e("LimpetVR-GL", "updateTexImage failed #$texFailCount", e) } }
                frameAvailable = false
            }
            // Decoder orientation (§4): honor the transform every frame so the
            // sampled image matches what the decoder produced.
            try { st.getTransformMatrix(texMat) } catch (_: Throwable) {}
        }

        head.getHeadView(headViewM, 0)

        if (testSweep != lastTestSweep) {
            lastTestSweep = testSweep
            if (testSweep) sweepT0 = android.os.SystemClock.elapsedRealtime()
            // The sweep composes a synthetic headView on top of the basis, so
            // the basis has to be the canonical one or the sweep axes come out
            // rotated by whatever pose the last recenter froze.
            System.arraycopy(basisShiftN, 0, basisShiftM, 0, 16)
        }
        if (testSweep) {
            val t = (android.os.SystemClock.elapsedRealtime() - sweepT0) / 1000f
            // yaw (about Y) ±12°/10s + pitch (about X) ±8°/7s + roll (about
            // view axis Z) ±15°/13s. Small on purpose: the panel stays in
            // frame so screenshots (or eyes) can judge each axis. Roll must
            // spin the panel IN PLACE (centered, tilted); yaw/pitch pan it.
            val yaw = 12f * kotlin.math.sin(2 * Math.PI.toFloat() * t / 10f)
            val pitch = 8f * kotlin.math.sin(2 * Math.PI.toFloat() * t / 7f + 1.3f)
            val roll = 15f * kotlin.math.sin(2 * Math.PI.toFloat() * t / 13f + 2.1f)
            // The sweep builds the device→world rotation; the view matrix
            // (world→head) is its transpose.
            Matrix.setIdentityM(tmpA, 0)
            Matrix.rotateM(tmpA, 0, yaw, 0f, 1f, 0f)
            Matrix.rotateM(tmpA, 0, pitch, 1f, 0f, 0f)
            Matrix.rotateM(tmpA, 0, roll, 0f, 0f, 1f)
            Matrix.transposeM(headViewM, 0, tmpA, 0)
        }

        if (recenterPending) {
            recenterPending = false
            // The test sweep drives a fixed canonical basis — never anchor it.
            if (!testSweep) {
                // Full app-side recenter. GVR's own call (gvr_recenter_tracking)
                // "resets the yaw to zero, leaving pitch and roll unmodified" —
                // that only ever fixes left/right, so the vertical placement
                // stayed glued to whatever it was. Fold the WHOLE current head
                // pose into the basis instead: app-forward/app-up become
                // wherever the head is now — yaw, pitch and roll (the same full
                // 3DOF snap the pre-GVR renderer did in computeEffLocked).
                val preFwd = lastEffFwd
                if (Matrix.invertM(tmpA, 0, headViewM, 0)) {
                    Matrix.multiplyMM(basisShiftM, 0, tmpA, 0, basisShiftN, 0)
                }
                snapCount++
                val p = Math.toDegrees(kotlin.math.asin(preFwd[1].coerceIn(-1f, 1f).toDouble()))
                val y = Math.toDegrees(kotlin.math.atan2(preFwd[0].toDouble(), -preFwd[2].toDouble()))
                FileLog.i("LimpetVR-basis", "recenter applied (snap=$snapCount tag=$snapTag " +
                    "yaw=${"%.1f".format(y)}° pitch=${"%.1f".format(p)}°)")
            }
        }

        synchronized(headViewM) {
            Matrix.multiplyMM(headWorldM, 0, headViewM, 0, basisShiftM, 0)
            if (!Matrix.invertM(invHeadWorldM, 0, headWorldM, 0)) Matrix.setIdentityM(invHeadWorldM, 0)
            // Gaze/tilt vectors live in WORLD space, so they come from the
            // INVERSE (head→world): forward = R·(0,0,-1), up = R·(0,1,0).
            val ihw = invHeadWorldM
            lastEffFwd = floatArrayOf(-ihw[8], -ihw[9], -ihw[10])
            lastEffUp = floatArrayOf(ihw[4], ihw[5], ihw[6])
        }

        // Screen-position sweep: run the window, then recenter — the app
        // basis absorbs the pose on the next frame (both axes).
        if (sweepRequest) {
            sweepRequest = false
            screenSweep = true
            screenSweepT0 = now()
            FileLog.i("LimpetVR-menu", "screenpos sweep start")
        }
        if (screenSweep && now() - screenSweepT0 >= SCREEN_SWEEP_MS) {
            screenSweep = false
            recenter("screenpos")
            FileLog.i("LimpetVR-menu", "screenpos sweep done -> recentered")
        }

        val cur = mode
        // Web page: upload the latest capture (if the page changed) before
        // anyone samples it.
        val wb = webBmpPending
        if (wb != null && webTexId >= 0) {
            webBmpPending = null
            try {
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, webTexId)
                GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, wb, 0)
                webConsumedFrames = ++webBmpConsumed
            } catch (_: Throwable) {
            }
        }
        // Dwell is suspended for the whole calibration sweep.
        if (!screenSweep) {
            when (cur) {
                Mode.BROWSER -> updateGaze()
                Mode.WEB -> {
                    // Every frame, panel open or not: the reticle falls back
                    // to this point when the gaze leaves the panel, and it
                    // was frozen while the panel was open because only the
                    // debug-gated path updated it.
                    updateWebPagePoint()
                    updateWebDebugPoint() // TEMP DEBUG: crosshair keeps tracking
                    // The tilt/hold update must run even while the panel is
                    // OPEN: it owns the open AND close transitions. Calling
                    // only updateGaze() when the panel was up meant the hold
                    // timer never decayed, so the close branch was
                    // unreachable - the panel stayed up until the X row was
                    // used, the dwell was never re-armed for the page, and
                    // the reticle had no live page point to fall back on.
                    updateWebPanelTilt()
                    if (webPanelOpen) updateGaze() else updateWebGaze()
                }
                Mode.VIDEO -> updateMenu()
            }
            if (cur != Mode.VIDEO) { menuOpen = false; menuHitValid = false }
        }
        // Panel fade (fast smoothstep both ways): browser fades with mode,
        // menu with menuOpen. Alphas gate the draw calls so a closing panel
        // keeps rendering until fully transparent.
        browAlpha = fadeAlpha(cur == Mode.BROWSER || (cur == Mode.WEB && webPanelOpen), true)
        menuAlpha = fadeAlpha(cur == Mode.VIDEO && menuOpen, false)

        // The curved cap bakes world size into the mesh, so curve/w/h join
        // the key while bent — curve 0 keeps the old unit-quad key (no churn
        // from size/aspect changes on the plain plane).
        val wantMeshKey = effProj().name + "|sh=" + shapingRevision.toString() + "|q=" + panoQuality +
            (if (mode == Mode.WEB) "|web=" + webPageW + "x" + webPageH else "") +
            if (effProj() == Projection.FLAT && screenCurve > 0f) {
                val (w, h) = screenDims()
                "|c=" + ((screenCurve * 100f) + 0.5f).toInt() +
                    "|w=" + ((w * 1000f) + 0.5f).toInt() + "|h=" + ((h * 1000f) + 0.5f).toInt()
            } else ""
        if (meshKey != wantMeshKey) { mesh = buildMesh(effProj()); meshKey = wantMeshKey }
    }

    private var browAlpha = 0f
    private var menuAlpha = 0f
    private val tmpFov = com.google.vr.sdk.base.FieldOfView()
    /** One-shot log of GVR's per-eye off-axis term (convergence diagnostics). */
    private val projLogged = booleanArrayOf(false, false)

    /** Per-eye matrices (§3): proj from GVR with the convergence trim baked
     *  into proj[8]; O = eyeView·N; ov = proj·O. pinVideo drops the head
     *  rotation (FLAT keeps the parallel eye shift, domes are head-locked).
     *
     *  Two projections, never mixed: projM is the UNSCALED eye frustum and
     *  drives every panel, the reticle/tooltip and the FLAT screen — fov+/-
     *  must not resize the UI. projDomeM is the panoramic projection and
     *  carries the FOV scale (§6.3); domeOvM = projDomeM·O feeds the dome
     *  and fisheye video only, with the same convergence trim so video and
     *  UI still fuse at one disparity. */
    private fun buildEyeMatrices(eye: Eye, physEye: Int) {
        System.arraycopy(eye.getPerspective(0.1f, 100f), 0, projM, 0, 16)
        // Convergence trim (§10): uniform clip-space offset per eye; polarity
        // matches the old physical frustum (left eye gets -ct). ADDED to
        // whatever GVR's off-axis frustum already carries — overwriting it
        // would throw away the viewer profile's physical convergence and
        // leave only the trim, which pushes the halves apart.
        val ct = convTrimNdc.coerceIn(-0.15f, 0.15f)
        val base8 = projM[8]
        val trim = if (physEye == 0) -ct else ct
        projM[8] = base8 + trim
        if (!projLogged[physEye]) {
            projLogged[physEye] = true
            val f = eye.fov
            FileLog.i("LimpetVR-GL", "proj eye=$physEye base8=${"%.4f".format(base8)} " +
                "trim=$ct fovL=${"%.1f".format(f.left)} fovR=${"%.1f".format(f.right)}")
        }
        // Panoramic projection: identical to projM at fovScale 1, otherwise
        // the eye FOV angles scale — trim re-applied so disparity matches.
        val fs = fovScale.coerceIn(0.5f, 1.5f)
        if (fs == 1f) {
            System.arraycopy(projM, 0, projDomeM, 0, 16)
        } else {
            val f = eye.fov
            tmpFov.setAngles(f.left * fs, f.right * fs, f.bottom * fs, f.top * fs)
            tmpFov.toPerspectiveMatrix(0.1f, 100f, projDomeM, 0)
            projDomeM[8] = projDomeM[8] + trim
        }

        Matrix.multiplyMM(eyeViewNM, 0, eye.eyeView, 0, basisShiftM, 0)
        if (pinVideo && mode == Mode.VIDEO) {
            if (projection == Projection.FLAT) {
                // Pinned flat screen keeps only the parallel eye shift:
                // eyeShift = eyeView · headView⁻¹ (a pure translation once
                // the head rotation cancels), then the basis shift.
                Matrix.invertM(tmpA, 0, headViewM, 0)
                Matrix.multiplyMM(tmpB, 0, eye.eyeView, 0, tmpA, 0)
                Matrix.setIdentityM(eyeShiftM, 0)
                eyeShiftM[12] = tmpB[12]; eyeShiftM[13] = tmpB[13]; eyeShiftM[14] = tmpB[14]
                Matrix.multiplyMM(tmpA, 0, eyeShiftM, 0, basisShiftM, 0)
                Matrix.multiplyMM(ovM, 0, projM, 0, tmpA, 0)
                Matrix.multiplyMM(domeOvM, 0, projDomeM, 0, tmpA, 0)
            } else {
                Matrix.multiplyMM(ovM, 0, projM, 0, basisShiftM, 0)
                Matrix.multiplyMM(domeOvM, 0, projDomeM, 0, basisShiftM, 0)
            }
        } else {
            Matrix.multiplyMM(ovM, 0, projM, 0, eyeViewNM, 0)
            Matrix.multiplyMM(domeOvM, 0, projDomeM, 0, eyeViewNM, 0)
        }
    }

    /** One eye: set this eye's viewport/scissor, clear, then the §3.3 draw
     *  order — video with depth on, panels/reticle/tooltip/toast with depth
     *  off, toast last. */
    private fun drawEye(eye: Eye) {
        val vp = eye.viewport
        vp.setGLViewport()
        vp.setGLScissor()
        GLES20.glEnable(GLES20.GL_SCISSOR_TEST)
        // GVR's own passes (clear/distortion) leave GL state as THEY like
        // it; premultiplied blend must be re-asserted every eye or the
        // panels and reticle paint their transparent texels opaque.
        blendAtEntry = GLES20.glIsEnabled(GLES20.GL_BLEND)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val physEye = if (eye.type == Eye.Type.RIGHT) 1 else 0
        // swapEyes: which texture half this eye samples (texcoords only).
        val uEye = if (swapEyes) 1 - physEye else physEye
        buildEyeMatrices(eye, physEye)

        val cur = mode
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        // Panels float over LIVE video in BROWSER mode too, so the queue
        // (and the draw) keep running there once frames have arrived.
        if (cur == Mode.WEB) { drawWeb(uEye); drawWebBar() }
        else if (cur == Mode.VIDEO || arrivedFrames > 0) drawVideo(uEye)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        if (cur == Mode.VIDEO) { if (menuAlpha > 0f) drawMenuPanel(menuAlpha) }
        else if (browAlpha > 0f) drawBrowser(browAlpha)
        drawHeadLocked()
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    /** Dwell progress 0..1 for the browser pointer (full ring → point).
     *  Rows, the X close button and the scroll strips each keep their OWN
     *  accumulator and drain the others while hovered, so the pointer has to
     *  read whichever one is live — checking browProgF alone left the ring at
     *  full size over the close buttons (and strips) even while they fired. */
    private fun browserDwellProg(): Float =
        maxOf(browProgF, xProgF, scrollTrigF).coerceIn(0f, 1f)

    /** Sweep progress 0..1 (reticle shrinks across the window). */
    private fun screenSweepProg(): Float =
        if (!screenSweep) 0f
        else ((now() - screenSweepT0).toFloat() / SCREEN_SWEEP_MS).coerceIn(0f, 1f)

    /** Dwell progress 0..1 for the menu pointer. */
    private fun menuDwellProg(): Float =
        if (menuHighlight != -2) menuProg[menuSlot(menuHighlight)].coerceIn(0f, 1f) else 0f

    private val clipV = FloatArray(4)

    /** Reticle + tooltip + toast: head-locked world quads drawn with ov. Any
     *  point on the head ray projects to the same screen point, so the
     *  reticle lands on the hovered hit no matter how deep it sits. */
    private fun drawHeadLocked() {
        val cur = mode
        // With the web panel up, the gaze is on its rows, so the reticle
        // animates with the PANEL's dwell (it was reading the page's, which
        // is always idle then, so rows never shrank).
        val panelGaze = cur == Mode.WEB && webPanelOpen
        val prog = if (screenSweep) screenSweepProg() else
            if (panelGaze) browserDwellProg() else
            if (cur == Mode.WEB) webDwellProg() else
            if (cur == Mode.BROWSER) browserDwellProg() else menuDwellProg()
        // The reticle is ALWAYS up in web mode: the page is the pointer.
        val show = testSweep || screenSweep || cur == Mode.WEB || cur == Mode.BROWSER ||
            (cur == Mode.VIDEO && menuOpen && (menuHitValid || prog > 0f))
        if (show) drawPointer(prog, reticleTexId, 1f)
        if (cur == Mode.VIDEO && menuOpen) reticleDiag(show, prog)
        if (cur == Mode.VIDEO && aimArmed) drawPointer(aimProg, aimTexId, 2f)
        // The pill belongs to the menu: follow its fade, and never outlive
        // it (tooltipVisible only clears when the gaze LEAVES the panel, so
        // gating on the flag alone left it hanging after the menu closed).
        if (cur == Mode.VIDEO && menuOpen && menuAlpha > 0f && tooltipVisible) drawTooltip()
        drawToast()
    }

    private var reticleDiagT = 0L
    /** Blend state GVR handed us at the top of the last eye draw. */
    private var blendAtEntry = true
    /** Throttled reticle health dump: is it drawn, where does it land,
     *  is the texture real. */
    private fun reticleDiag(show: Boolean, prog: Float) {
        if (now() - reticleDiagT < 500L) return
        reticleDiagT = now()
        val d = panelDistM
        headLockedAt(tmpA, 0f, 0f, -d)
        Matrix.multiplyMM(mvpM, 0, ovM, 0, tmpB, 0)
        Matrix.multiplyMV(clipV, 0, mvpM, 0, floatArrayOf(0f, 0f, -d, 1f), 0)
        val w = clipV[3]
        val s = kotlin.math.sin(RETICLE_ANG) * d * (1f - 0.85f * prog.coerceIn(0f, 1f))
        FileLog.i("LimpetVR-GL", "ret show=$show prog=$prog hl=$menuHighlight hit=$menuHitValid " +
            "s=${"%.4f".format(s)} d=$d tex=$reticleTexId isTex=${GLES20.glIsTexture(reticleTexId)} " +
            "ndc=${"%.3f".format(clipV[0] / w)},${"%.3f".format(clipV[1] / w)} w=${"%.2f".format(w)} " +
            "alpha=${"%.2f".format(menuAlpha)} blendIn=$blendAtEntry")
    }

    /** Head-locked ring at panel depth, shrinking (1 - 0.85·prog) to a point.
     *  Placed in HEAD space (invHeadWorldM⁻¹·ov) so it rides the gaze ray:
     *  ov alone would pin it to a fixed world point, which lands off-screen
     *  as soon as the head tilts to the menu. */
    private fun drawPointer(prog: Float, texId: Int, sizeMul: Float) {
        // Over the page the reticle belongs at the page's depth, not the
        // panel's: a near reticle over a far screen is offset per eye.
        // Until the first page hit (or after the gaze leaves it) fall back to
        // the screen centre's distance, so the reticle never hops between the
        // page depth and the panel depth.
        if (mode == Mode.WEB && !webPanelOpen && webPagePointOk) {
            // On the page: sit exactly on the sampled surface point, so the
            // reticle and the page converge at the same distance in each
            // eye. Placing it "d metres along the ray" left the stereo pair
            // slightly wider than the page it was pointing at.
            // ON the page point where the gaze ray lands. That point is
            // world-anchored, which is correct for a cursor on a flat
            // screen: it moves as the head moves because the ray genuinely
            // sweeps across the page. Deriving the position from the ray
            // instead (head-locked at a fixed distance) put the reticle on
            // a rigid stalk that slid the opposite way.
            webReticleAt(webPagePoint)
            putQuad(ptrVerts, ptrTex, -webRetS(sizeMul, prog), -webRetS(sizeMul, prog),
                webRetS(sizeMul, prog), webRetS(sizeMul, prog))
            Matrix.multiplyMM(mvpM, 0, ovM, 0, tmpA, 0)
            drawQuadTex(texId, mvpM)
            return
        }
        // The panel floats at browserElevDeg ABOVE eye level, so the reticle
        // has to sit on the same point of the sphere - placing it dead ahead
        // at eye level put it below the panel, which is where the gaze lands
        // when you look up at the menu. (It is drawn over the panel whether or
        // not the panel is open, so this also covers the web-panel case.)
        val d = panelDistM
        // Off the panel: fall back to the page if the gaze is still on it,
        // otherwise hold the last known point rather than snapping to a
        // centre that is nowhere near what the user is looking at.
        // Order matters. While the panel is open the gaze belongs to the
        // panel, so the panel wins; only when the ray misses the panel does
        // the reticle fall back to the page. Putting the panel's own hit
        // ahead of the page point (both are panel data while it is open)
        // meant the reticle could never leave the panel at all.
        // In the gap (above the panel, or off its edge) the ray still has to
        // drive the reticle. Holding the last position instead froze it at a
        // fixed spot - the same spot every time, and whichever spot the gaze
        // last crossed - which is what it did above the menu. Ray-following
        // is continuous through the gap, so it slides on and off the panel.
        val onPanel = webPanelPoint(ptrTmp3)
        if (!onPanel && !webPagePointOk) {
            webReticleRay(d)
            val sr = webRetS(sizeMul, prog)
            putQuad(ptrVerts, ptrTex, -sr, -sr, sr, sr)
            Matrix.multiplyMM(mvpM, 0, ovM, 0, tmpA, 0)
            drawQuadTex(texId, mvpM)
            return
        }
        val tgt = when {
            onPanel -> ptrTmp3
            webPagePointOk -> webPagePoint
            else -> webLastPoint
        }
        webReticleAt(tgt)
        val sr = webRetS(sizeMul, prog)
        putQuad(ptrVerts, ptrTex, -sr, -sr, sr, sr)
        Matrix.multiplyMM(mvpM, 0, ovM, 0, tmpA, 0)
        drawQuadTex(texId, mvpM)
    }

    /** Head-to-screen-centre distance, the reticle's depth before the gaze
     *  has landed on the page. 0 if the screen is not there. */
    private fun webScreenCentreDist(): Float {
        webPointAt(0.5f, 0.5f, wpTmp)
        val dx = wpTmp[0] - invHeadWorldM[12]
        val dy = wpTmp[1] - invHeadWorldM[13]
        val dz = wpTmp[2] - invHeadWorldM[14]
        return kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
    }

    /** tmpB = invHeadWorldM · T(x,y,z): a head-space offset in world coords,
     *  ready to premultiply by ovM. */
    private fun headLockedAt(t: FloatArray, x: Float, y: Float, z: Float) {
        Matrix.setIdentityM(t, 0)
        Matrix.translateM(t, 0, x, y, z)
        // ovM is world->clip, so the point handed to it must be in WORLD
        // space: headWorldM (head->world) maps the head-local offset out to
        // the world. Composing invHeadWorldM (world->head) here applied the
        // head pose twice, which is a no-op only while the pose is identity
        // and a wrong depth/rotation everywhere else - so head-locked items
        // sat at the right screen spot but the wrong distance in each eye,
        // and recenter (which writes a real rotation into the basis) is
        // exactly when it started to show.
        Matrix.multiplyMM(tmpB, 0, headWorldM, 0, t, 0)
    }


    /** Gaze tooltip pill, just under the reticle. Half-extents and offset
     *  are design units × menuScale() so the pill keeps its apparent size
     *  at every panel distance (same rule as the menu rect). */
    private fun drawTooltip() {
        maybeUploadTooltip()
        if (!tooltipVisible) return
        val k = menuScale()
        putQuad(ptrVerts, ptrTex, -TIP_HALF_W * k, -TIP_HALF_H * k, TIP_HALF_W * k, TIP_HALF_H * k)
        headLockedAt(tmpA, 0f, -1.0f * k, -panelDistM)
        Matrix.multiplyMM(mvpM, 0, ovM, 0, tmpB, 0)
        drawQuadTex(tipTexId, mvpM, menuAlpha)
    }

    /** 512×96 pill bearing the current tooltip text; re-uploaded only when
     *  the text changes (cache hit per frame otherwise). */
    private fun maybeUploadTooltip() {
        val txt = tooltipText
        if (txt == lastTipText && tipBitmap != null) return
        lastTipText = txt
        val W = 512; val H = 96
        val bmp = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.TRANSPARENT)
        val p = Paint(Paint.ANTI_ALIAS_FLAG)
        p.color = Color.argb(224, 10, 14, 22)
        c.drawRoundRect(2f, 2f, (W - 2).toFloat(), (H - 2).toFloat(), 20f, 20f, p)
        p.color = Color.rgb(125, 211, 252)
        p.style = Paint.Style.STROKE; p.strokeWidth = 3f
        c.drawRoundRect(2f, 2f, (W - 2).toFloat(), (H - 2).toFloat(), 20f, 20f, p)
        p.style = Paint.Style.FILL
        p.color = Color.WHITE; p.textSize = 44f; p.textAlign = Paint.Align.CENTER
        val t = txt.take(28)
        var ts = 44f
        while (ts > 16f && p.measureText(t) > W - 40f) { ts -= 2f; p.textSize = ts }
        c.drawText(t, W / 2f, H / 2f + ts * 0.35f, p)
        p.textAlign = Paint.Align.LEFT
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tipTexId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        tipBitmap?.recycle()
        tipBitmap = bmp
    }

    /** One textured quad (ptrVerts/ptrTex, world coords) through prog2d. */
    private fun drawQuadTex(texId: Int, mvp: FloatArray, alpha: Float = 1f) {
        GLES20.glUseProgram(prog2d)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId)
        GLES20.glUniform1i(uTex2d, 0)
        GLES20.glUniform1f(uAlpha2d, alpha.coerceIn(0f, 1f))
        GLES20.glUniformMatrix4fv(uMvp2d, 1, false, mvp, 0)
        GLES20.glEnableVertexAttribArray(aPos2d)
        GLES20.glVertexAttribPointer(aPos2d, 3, GLES20.GL_FLOAT, false, 0, ptrVerts)
        GLES20.glEnableVertexAttribArray(aTex2d)
        GLES20.glVertexAttribPointer(aTex2d, 2, GLES20.GL_FLOAT, false, 0, ptrTex)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos2d)
        GLES20.glDisableVertexAttribArray(aTex2d)
    }

    // scratch buffers: zero per-frame allocation on the GL thread (direct
    // ByteBuffer churn caused GC strobes that read as pointer flicker)
    private val v4 = FloatArray(4)
    private val ptrVerts: FloatBuffer = ByteBuffer.allocateDirect(12 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val ptrTex: FloatBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private fun putQuad(vb: FloatBuffer, tb: FloatBuffer, x0: Float, y0: Float, x1: Float, y1: Float) {
        vb.clear()
        vb.put(x0); vb.put(y1); vb.put(0f)
        vb.put(x0); vb.put(y0); vb.put(0f)
        vb.put(x1); vb.put(y1); vb.put(0f)
        vb.put(x1); vb.put(y0); vb.put(0f)
        vb.position(0)
        tb.clear()
        tb.put(0f); tb.put(0f)
        tb.put(0f); tb.put(1f)
        tb.put(1f); tb.put(0f)
        tb.put(1f); tb.put(1f)
        tb.position(0)
    }

    /** Zoom number z for the finite FLAT quad: texture magnification about
     *  the half-image center (the flat quad keeps texture zoom, §5). Domes
     *  slide their sphere toward the viewer instead and upload 1 here. */
    private fun flatZoomF(): Float = zoom.coerceIn(0.1f, 20f)

    /** Texture zoom for the web page: always 1.
     *
     *  The page capture is 16:9 and screenDims() builds a 16:9 screen for
     *  it, so the texture covers the surface exactly - there is never
     *  anything to crop, and magnifying the page is what screenSize does:
     *  the capture is stretched across the screen, so a larger screen
     *  shows the SAME whole page, bigger. Cropping is therefore redundant
     *  for the page, and it is the video's own control (settings.videoZoom)
     *  which the web panel used to overwrite, changing the 2D video zoom
     *  behind the user's back. */
    private fun webZoomF(): Float = 1f

    // ---------- gaze ----------
    // Panel size grows sub-linearly with distance (sqrt): distance changes
    // stay clearly visible (nearer = bigger) while extremes stay comfortable.
    // A linear width (= constant on-screen size) hid the control entirely.
    // Matches stock 0.96 half-width at the 2.4 m default.
    private fun panelHalfW(): Float = 0.62f * kotlin.math.sqrt(panelDistM)
    private fun panelHalfH(): Float = panelHalfW() * 0.62f

    // last ray↔panel hit in panel-world coords (for the at-depth cursor)
    private var hitX = 0f
    private var hitY = 0f
    private var hitZ = -2.4f
    private var hitValid = false
    // hovered slider fraction (panel-wide u) for the live tooltip; -1 = none
    private var sliderHoverU = -1f
    // gaze u where the last slider dwell fired; sliding the gaze along the
    // bar re-arms shrink + fire without leaving the row (NaN = disarmed)
    private var lastFiredU = Float.NaN
    // head-forward at the last slider fire: re-arm requires the HEAD to have
    // moved, so a panel resize shifting the mapping under a steady gaze
    // can't trigger a spurious feedback-loop refire
    private var lastFiredFwd = floatArrayOf(0f, 0f, -1f)
    // stillness gate state: dwell must NEVER fire while the head is moving,
    // or slow test turns sweep the gaze across rows and trigger accidental
    // navigation + recentering mid-turn (reads as rotation/pan glitches).
    private val dwellPrevM = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
    private var dwellPrevT = 0L
    private var dwellPrevInit = false

    private fun updateGaze() {
        val rows = browserRows
        if (rows.isEmpty()) { highlight = -1; hitValid = false; inXZone = false; xDwellFired = false;
            sideBtnDir = 0; sideBtnFired = false; sideBtnProg = 0f; scrollTrigF = 0f; sliderHoverU = -1f; lastFiredU = Float.NaN; return }
        if (now() < inputGraceUntil) { dwellStart = now(); sliderHoverU = -1f; return }
        // Windowed stillness + leaky dwell (same as the play menu): jitter
        // and row churn only dent progress instead of zeroing the timer.
        val nowMs = now()
        val still = synchronized(headViewM) { browserStill.update(headViewM, nowMs) }
        val dtMs = (nowMs - browProgT).coerceIn(0L, 500L)
        browProgT = nowMs
        if (!still) browProgF = maxOf(0f, browProgF - dtMs / 600f)
        // head-forward ray in world (panel floats at browserElevDeg,
        // facing the viewer; elevation 0 = centered at z=-D, identical math).
        // Uses the recentered orientation, so the reticle and the hover agree.
        val o = invHeadWorldM        // head position in world
        val fwd = lastEffFwd
        val fx = fwd[0]; val fy = fwd[1]; val fz = fwd[2]
        val d = panelDistM
        val el = Math.toRadians(browserElevDeg.toDouble()).toFloat()
        val cx = 0f; val cy = (kotlin.math.sin(el) * d); val cz = (-kotlin.math.cos(el) * d)
        var pnx = -cx; var pny = -cy; var pnz = -cz
        val pnl = kotlin.math.sqrt(pnx * pnx + pny * pny + pnz * pnz).coerceAtLeast(1e-6f)
        pnx /= pnl; pny /= pnl; pnz /= pnl
        val denom = fx * pnx + fy * pny + fz * pnz
        if (denom < -0.05f) {
            val t = ((cx - o[12]) * pnx + (cy - o[13]) * pny + (cz - o[14]) * pnz) / denom
            val hx = o[12] + fx * t; val hy = o[13] + fy * t; val hz = o[14] + fz * t
            // panel coords: right = (1,0,0); up = n × right = (0, nz, -ny)
            val upx = 0f; val upy = pnz; val upz = -pny
            val upl = kotlin.math.sqrt(upy * upy + upz * upz).coerceAtLeast(1e-6f)
            val hw = panelHalfW(); val hh = panelHalfH()
            val alongRight = hx - cx
            val alongUp = ((hx - cx) * upx + (hy - cy) * (upy / upl) + (hz - cz) * (upz / upl))
            if (alongRight >= -hw && alongRight <= hw && alongUp >= -hh && alongUp <= hh) {
                hitX = hx; hitY = hy; hitZ = hz; hitValid = true
                val u = (alongRight + hw) / (2 * hw)
                val v = (hh - alongUp) / (2 * hh)
                // PgUp / PgDn, web panel only. Checked before the rows so a
                // gaze on a button never selects the row underneath it.
                if (webSideBtns) {
                    val bx = u * TEX
                    val by = v * TEX
                    val onUp = bx >= SIDE_BTN_X0 && bx <= SIDE_BTN_X1 &&
                        by >= SIDE_BTN_UP_Y0 && by < SIDE_BTN_UP_Y0 + SIDE_BTN_H
                    val onDn = bx >= SIDE_BTN_X0 && bx <= SIDE_BTN_X1 &&
                        by >= SIDE_BTN_DN_Y0 && by < SIDE_BTN_DN_Y0 + SIDE_BTN_H
                    if (onUp || onDn) {
                        val dir = if (onUp) -1 else 1
                        sliderHoverU = -1f
                        lastFiredU = Float.NaN
                        fireBlockedLogged = false
                        highlight = -1; dwellFiredFor = -2
                        browProgF = maxOf(0f, browProgF - dtMs / 600f)
                        scrollEngage = 0; scrollTrigDir = 0; scrollTrigF = 0f
                        if (sideBtnDir != dir) { sideBtnDir = dir; sideBtnProg = 0f; sideBtnFired = false }
                        if (!sideBtnFired) {
                            if (still) sideBtnProg += dtMs / dwellMs.toFloat()
                            else sideBtnProg = maxOf(0f, sideBtnProg - dtMs / 600f)
                        }
                        if (still && !sideBtnFired && sideBtnProg >= 1f) {
                            sideBtnFired = true
                            sideBtnProg = 1f
                            FileLog.i("LimpetVR-web", "FIRE ${if (dir < 0) "PageUp" else "PageDown"}")
                            onWebEvent(WebEvent.PageKey(dir))
                        }
                        return
                    }
                    sideBtnDir = 0; sideBtnFired = false
                    sideBtnProg = maxOf(0f, sideBtnProg - dtMs / 600f)
                }
                // No close button on the panel: the gaze is still inside the
                // panel's visibility range, so closing it just made it
                // spring open again. It closes on its own when the gaze
                // returns to the page.
                val pin = pinTopRows.coerceIn(0, 2)
                var idx = -1
                // Strips mode on file pages always, and on settings pages
                // while testing once the list overflows the plain window
                // (no pinned rows there).
                val stripsOn = pin > 0 || rows.size > VISIBLE_ROWS
                if (stripsOn) {
                    // File pages: strips + pinned rows + fractional window.
                    val win = winRows(pin)
                    val ypx = v * TEX
                    val upY0 = upStripY0(pin); val rY0 = rowsY0(pin); val dnY0 = downStripY0(pin)
                    val scrollable = rows.size > pin + win
                    val inBarX = u * TEX >= 20f && u * TEX <= 1004f
                    if (scrollable && inBarX && ypx >= upY0 && ypx < upY0 + STRIP_H) {
                        stripGaze(-1, dtMs, still, hx, hy, hz)
                        return
                    }
                    if (scrollable && inBarX && ypx >= dnY0 && ypx < dnY0 + STRIP_H) {
                        stripGaze(1, dtMs, still, hx, hy, hz)
                        return
                    }
                    if (scrollEngage != 0 || scrollTrigDir != 0) {
                        scrollEngage = 0; scrollTrigDir = 0; scrollTrigF = 0f
                    }
                    if (ypx >= PIN_Y0 && ypx < PIN_Y0 + pin * ROW_H) {
                        idx = ((ypx - PIN_Y0) / ROW_H).toInt().coerceIn(0, pin - 1)
                    } else if (ypx >= rY0 && ypx < rY0 + win * ROW_H && rows.size > pin) {
                        val f = pin + scrollPos + (ypx - rY0) / ROW_H
                        idx = f.toInt().coerceIn(pin, rows.size - 1)
                    }
                } else {
                    // Settings pages: plain fixed window, no strips.
                    val vi = ((v * TEX - ROWS_Y0) / ROW_H).toInt()
                    if (vi in 0 until VISIBLE_ROWS) {
                        idx = (scrollPos.toInt() + vi).coerceIn(0, rows.size - 1)
                    }
                }
                if (idx in rows.indices) {
                    // dead rows are gaze rest zones: drain, never accumulate or fire
                    if (rows[idx].dead) {
                        if (idx != highlight) { highlight = idx; dwellFiredFor = -2 }
                        sliderHoverU = -1f
                        browProgF = maxOf(0f, browProgF - dtMs / 600f)
                        return
                    }
                    sliderHoverU = if (rows[idx].slideKey != null || rows[idx].segActions.isNotEmpty()) u else -1f
                    if (idx != highlight) {
                        highlight = idx; dwellFiredFor = -2
                        lastFiredU = Float.NaN
                        fireBlockedLogged = false
                        // small credit on row change (replaces the old
                        // 150ms-style stability delay): keeps flips cheap
                        browProgF = minOf(browProgF, 0.25f)
                    }
                    // Slider re-arm: sliding the gaze along the bar after a
                    // fire restarts shrink + fire without leaving the row.
                    // Requires real head motion: a panel resize shifting the
                    // mapping under a steady gaze must not refire by itself.
                    // (Gaze is head-driven — no eye tracking — so any genuine
                    // slide moves the head; jitter stays far below both gates.)
                    if (highlight == dwellFiredFor && rows[idx].slideKey != null &&
                        !lastFiredU.isNaN() && kotlin.math.abs(u - lastFiredU) > 0.05f &&
                        angleDeg(lastFiredFwd, lastEffFwd) > 2f) {
                        dwellFiredFor = -2
                        browProgF = 0f
                        lastFiredU = Float.NaN
                    }
                    // Accumulate only until fired for this hover; after firing
                    // hold the shrunken state (no second shrink animation).
                    // Re-entry (off-panel resets dwellFiredFor) restarts it.
                    if (highlight != dwellFiredFor) {
                        if (still) browProgF += dtMs / dwellMs.toFloat()
                        else browProgF = maxOf(0f, browProgF - dtMs / 600f)
                    }
                    if (still && highlight != dwellFiredFor && browProgF >= 1f) {
                        dwellFiredFor = highlight
                        browProgF = 1f
                        // slider rows pass the bar-mapped fraction so one dwell
                        // sets any value; segmented rows pass panel-wide u for
                        // segment picking; plain rows null
                        val frac = if (rows[idx].slideKey != null) barFrac(u)
                            else if (rows[idx].segActions.isNotEmpty()) u else null
                        if (rows[idx].slideKey != null) {
                            lastFiredU = u
                            lastFiredFwd = lastEffFwd.clone()
                        }
                        FileLog.i("LimpetVR-browser", "FIRE row=$idx frac=$frac")
                        onBrowserActivate(highlight, frac)
                        return
                    } else if (still && highlight == dwellFiredFor && browProgF >= 1f && !fireBlockedLogged) {
                        fireBlockedLogged = true
                        FileLog.i("LimpetVR-browser", "BLOCKED repeat dwell row=$highlight prog=$browProgF")
                    }
                    return
                }
            }
        }
        // not hovering the panel: hide cursor, drain progress (no zeroing).
        // Reset the row-fired latch so looking back re-arms shrink + fire.
        hitValid = false
        inXZone = false
        xDwellFired = false
        sideBtnDir = 0; sideBtnFired = false
        sideBtnProg = maxOf(0f, sideBtnProg - 16f / 600f)
        dwellFiredFor = -2
        lastFiredU = Float.NaN
        fireBlockedLogged = false
        sliderHoverU = -1f
        xProgF = maxOf(0f, xProgF - 16f / 600f)
        browProgF = maxOf(0f, browProgF - 16f / 600f)
        scrollTrigF = maxOf(0f, scrollTrigF - 16f / 600f)
    }

    /** Gaze on a scroll strip (file pages only): a short still-gaze
     *  engages it, then the rows window glides continuously while the gaze
     *  stays on the strip. Looking away stops immediately. */
    private fun stripGaze(dir: Int, dtMs: Long, still: Boolean, hx: Float, hy: Float, hz: Float) {
        hitX = hx; hitY = hy; hitZ = hz; hitValid = true
        highlight = -1; dwellFiredFor = -2
        sliderHoverU = -1f; lastFiredU = Float.NaN; fireBlockedLogged = false
        inXZone = false; xDwellFired = false
        browProgF = maxOf(0f, browProgF - dtMs / 600f)
        xProgF = maxOf(0f, xProgF - dtMs / 600f)
        if (scrollEngage == dir) {
            val pin = pinTopRows.coerceIn(0, 2)
            val maxS = maxOf(0, browserRows.size - pin - winRows(pin)).toFloat()
            scrollPos = (scrollPos + dir * STRIP_ROWS_PER_SEC * dtMs / 1000f).coerceIn(0f, maxS)
            return
        }
        if (scrollTrigDir != dir) { scrollTrigDir = dir; scrollTrigF = 0f }
        if (still) scrollTrigF += dtMs / STRIP_TRIG_MS
        else scrollTrigF = maxOf(0f, scrollTrigF - dtMs / 600f)
        if (scrollTrigF >= 1f) { scrollTrigF = 1f; scrollEngage = dir }
    }

    /** Panel open/close fade state (GL thread). */
    private var browFadeVis = false
    private var browFadeT0 = 0L
    private var menuFadeVis = false
    private var menuFadeT0 = 0L

    /** Fast fade alpha for a panel: smoothstep over PANEL_FADE_MS after
     *  each visibility transition. `which=true` = browser, false = menu. */
    private fun fadeAlpha(want: Boolean, which: Boolean): Float {
        val t0: Long
        if (which) {
            if (want != browFadeVis) { browFadeVis = want; browFadeT0 = now() }
            t0 = browFadeT0
        } else {
            if (want != menuFadeVis) { menuFadeVis = want; menuFadeT0 = now() }
            t0 = menuFadeT0
        }
        val t = ((now() - t0).toFloat() / PANEL_FADE_MS).coerceIn(0f, 1f)
        val s = t * t * (3f - 2f * t)
        return if ((if (which) browFadeVis else menuFadeVis)) s else 1f - s
    }

    private fun ensureVisible() {
        val pin = pinTopRows.coerceIn(0, 2)
        val stripsOn = pin > 0 || browserRows.size > VISIBLE_ROWS
        val win = if (stripsOn) winRows(pin) else VISIBLE_ROWS
        val maxS = maxOf(0, browserRows.size - pin - win).toFloat()
        scrollPos = scrollPos.coerceIn(0f, maxS)
        val h = highlight
        if (h < 0 || h >= browserRows.size) return
        if (h < pin) return // pinned row: always visible, never scrolls
        val base = scrollPos.toInt()
        if (h < pin + base) scrollPos = (h - pin).toFloat()
        else if (h >= pin + base + win) scrollPos = (h - pin - win + 1).toFloat()
        scrollPos = scrollPos.coerceIn(0f, maxS)
    }

    private fun now() = System.currentTimeMillis()

    /** Angle between two head-forward vectors, degrees. */
    private fun angleDeg(a: FloatArray, b: FloatArray): Float {
        val dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).coerceIn(-1f, 1f)
        return Math.toDegrees(kotlin.math.acos(dot).toDouble()).toFloat()
    }

    /** Panel-wide gaze fraction -> slider-bar fraction (bar spans x 44..1000 of TEX 1024). */
    private fun barFrac(u: Float) = ((u * TEX - 44f) / 956f).coerceIn(0f, 1f)

    /** Design-space panel x -> seek fraction, from the bar's own rect. */
    private fun seekFrac(x: Float) =
        (((x - menuBar.x) / menuBar.hw) * 0.5f + 0.5f).coerceIn(0f, 1f)

    // ---------- play menu ----------
    // World-locked panel floating up (or down) in the recentered frame.
    // Look past the side's trigger angle to open; the close band sits 25° lower.
    // The pointer stays hidden during video until the menu opens.
    /** Windowed stillness gate: displacement over the trailing ~250ms.
     *  Per-frame deltas are useless — game-RV jitter trips a per-frame
     *  gate ~30×/s (see the menu motion-reset bursts in the log), so
     *  dwell can only accumulate during lucky-still streaks. Over a
     *  window, zero-mean noise cancels while real motion accumulates. */
    private class MotionStillness(private val windowMs: Long, private val limitDeg: Float) {
        private val refM = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
        private var refT = 0L
        private var init = false
        private val tmpA = FloatArray(16)
        private val tmpB = FloatArray(16)
        /** Call on the GL thread with the live rotation matrix. */
        fun update(raw: FloatArray, nowMs: Long): Boolean {
            if (!init) {
                System.arraycopy(raw, 0, refM, 0, 16)
                refT = nowMs
                init = true
                return true
            }
            Matrix.transposeM(tmpA, 0, refM, 0)
            Matrix.multiplyMM(tmpB, 0, tmpA, 0, raw, 0)
            val tr = tmpB[0] + tmpB[5] + tmpB[10]
            val ang = Math.toDegrees(
                kotlin.math.acos(((tr - 1f) / 2f).toDouble().coerceIn(-1.0, 1.0))
            ).toFloat()
            if (nowMs - refT >= windowMs) {
                System.arraycopy(raw, 0, refM, 0, 16)
                refT = nowMs
            }
            return ang < limitDeg
        }
    }
    private val menuStill = MotionStillness(250L, 4f)
    private val browserStill = MotionStillness(250L, 2.5f)
    // leaky dwell integrators: progress grows while still on target and
    // drains slowly otherwise. Resets can never win against tremor or
    // target churn — flicker only dents progress instead of zeroing it.
    private val menuProg = FloatArray(19)
    private var menuProgT = 0L
    private var browProgF = 0f
    private var browProgT = 0L
    private fun menuSlot(id: Int) = if (id == -1) 18 else id
    private fun menuUsable(id: Int) = id != -2

    /** A gaze target on the play-menu panel, in DESIGN panel space (the
     *  layout authored for a MENU_DESIGN_R viewing distance; menuScale()
     *  maps it onto the live panel distance). id: the MenuEvent.Press idx
     *  (-1 = seek bar, -2 = decorative, never fired). */
    private data class MenuBtn(
        val id: Int, val x: Float, val y: Float,
        val hw: Float = MENU_BTN_HALF, val hh: Float = MENU_BTN_HALF,
        val glyph: String = ""
    )

    /** Panel-local names, indexed by MenuEvent.Press idx (tooltips). */
    private val MENU_NAMES = arrayOf(
        "settings", "shape", "files", "prev", "rew", "play",
        "ff", "next", "zoom+", "zoom−", "vol+", "vol−",
        "flip", "recenter", "fov−", "fov+", "", ""
    )

    /** The whole play menu: transport row on the bar's centreline, the seek
     *  bar under it (shifted left), the zoom/fov/volume pairs as three
     *  columns at the right (+ just above −, both on the seek-bar band),
     *  recenter top left and flip top right (both level with the title
     *  strip; the pane's top edge hugs them so the top band stays tight).
     *  Title and backdrop are decorative. */
    private val menuButtons = arrayOf(
    // Transport row runs one pitch further left now that web sits beside
    // the files button (9 buttons, still clear of the zoom/fov/vol columns).
    MenuBtn(0, -4.50f, 0f, glyph = "⚙"),   // settings, ahead of shape
    MenuBtn(1, -3.75f, 0f, glyph = "⧗"),
    MenuBtn(2, -3.00f, 0f, glyph = "📁"),
    MenuBtn(18, -2.25f, 0f, glyph = "🌐"),  // web, beside files
    MenuBtn(3, -1.50f, 0f, glyph = "⏮"),
        MenuBtn(4, -0.75f, 0f, glyph = "⏪"),
        MenuBtn(5, 0.00f, 0f),   // play / pause drawn from menuPlaying
        MenuBtn(6, 0.75f, 0f, glyph = "⏩"),
        MenuBtn(7, 1.50f, 0f, glyph = "⏭"),
        // adjustment columns down the right side, left to right
        // zoom / fov / volume, + paired just above − on the seek-bar line;
        // volume is a third the size of the others
        MenuBtn(8, 3.15f, -0.25f, 0.15f, 0.15f),   // zoom+
        MenuBtn(9, 3.15f, -0.75f, 0.15f, 0.15f),  // zoom−
        MenuBtn(15, 3.90f, -0.25f, 0.15f, 0.15f),  // fov+
        MenuBtn(14, 3.90f, -0.75f, 0.15f, 0.15f), // fov−
        MenuBtn(10, 4.65f, -0.25f, 0.05f, 0.05f),  // vol+
        MenuBtn(11, 4.65f, -0.75f, 0.05f, 0.05f), // vol−
        MenuBtn(12, 4.65f, 0.70f),  // flip: ⇅ top right, atop vol+ column
        MenuBtn(13, -4.50f, 0.70f)                // recenter, top left
    )
    /** Seek bar (id -1), decorative title and backdrop, all design units.
     *  The bar sits left of centre so the − row of the ± columns has its
     *  own lane at the right edge. */
    private val menuBar = MenuBtn(-1, -0.45f, -0.75f, 3.0f, 0.3f)
    private val menuTitleRect = MenuBtn(-2, 0f, 0.70f, 2.55f, 0.3f)
    /** Pane: bottom stays at −1.2; top hugs the title row (1.05) so the
     *  top band uses less vertical space than the old symmetric 1.2. */
    private val menuBackdrop = MenuBtn(-2, 0f, -0.075f, 5.1f, 1.125f)

    /** Effective open angle (always 10..60) and side from the ⇅ toggle. */
    private fun menuOpenAngle(): Float =
        if (menuSideUp) menuAngleUp.coerceIn(10f, 60f)
        else kotlin.math.abs(menuAngleDown).coerceIn(10f, 60f)
    private fun menuIsBelow(): Boolean = !menuSideUp
    /** Menu panel elevation (deg): the whole panel floats past the open
     *  angle so looking at its NEAREST edge keeps you beyond the trigger.
     *  That edge sits MENU_MARGIN_DEG past the trigger, and its angular
     *  offset from the panel centre is atan(nearExtent / MENU_DESIGN_R) —
     *  constant, because menuScale() shrinks the panel with the distance
     *  (the design rect is authored for MENU_DESIGN_R). 6.7° reproduces the
     *  old bottom-edge clearance bit-for-bit (40° open → 55.3°, old 54.5°).
     *  No hysteresis anywhere: open at/above the angle, closed below it. */
    private fun menuElevDeg(): Float {
        // side up: the near edge is the panel's BOTTOM; side down: its TOP.
        val near = if (menuIsBelow()) MENU_EXT_TOP else MENU_EXT_BOT
        val extDeg = Math.toDegrees(kotlin.math.atan(near / MENU_DESIGN_R).toDouble()).toFloat()
        return ((menuOpenAngle() + extDeg + MENU_MARGIN_DEG).coerceAtMost(85f)) *
            (if (menuIsBelow()) -1f else 1f)
    }
    /** Animated elevation for the ⇅ flip: smooth sweep through the middle,
     *  settling on the target side. */
    fun menuElevCurrent(): Float {
        val target = menuElevDeg()
        val dt = now() - menuAnimT0
        if (dt >= menuAnimMs) return target
        val t = (dt.toFloat() / menuAnimMs).coerceIn(0f, 1f)
        val s = t * t * (3f - 2f * t)
        return menuAnimFrom + (target - menuAnimFrom) * s
    }
    /** The design rect is authored for a MENU_DESIGN_R viewing distance —
     *  the same 11.95 m the flat screen sits at — so scaling the WHOLE
     *  rect by panelDistM/MENU_DESIGN_R makes the panel's APPARENT size
     *  constant (±5.3 at any depth = 47.9° across; the old 0.62·√d panel
     *  measured 43.6° at the 2.4 m default and shrank as you moved it)
     *  while the slider still changes the panel's depth (real stereo
     *  parallax). The extra ~4° is the right-margin column the old
     *  transport-only panel didn't have. */
    private fun menuScale(): Float = panelDistM.coerceIn(0.5f, 20f) / MENU_DESIGN_R

    // ---------- play menu ----------

    /** Circle-to-recenter gesture master switch (2D settings, default on). */
    @Volatile var circleGestureEnabled = true
    /** UI-thread request: close the menu and arm the aim pointer (GL consumes). */
    @Volatile private var aimRequest = false
    /** Aim pointer live. Volatile: armed/cleared on GL, read on UI (tap cancels). */
    @Volatile private var aimArmed = false
    // GL-thread aim dwell state (head-locked big blue pointer, 2x dwell)
    private var aimProg = 0f
    private var aimT = 0L
    private var aimArmedAt = 0L
    private val aimW = FloatArray(3)
    // circle detector: fixed ring buffer, zero per-frame allocation
    // (160 slots ≈ 5.3s at the 33ms sample step: the window must be
    // longer than the gesture, or the first loop ages out mid-draw)
    private val circT = LongArray(160)
    private val circX = FloatArray(160)
    private val circY = FloatArray(160)
    private var circHead = 0
    private var circN = 0
    private var circLastT = 0L
    private var circCooldownUntil = 0L
    private var circNearT = 0L

    /** Dwell the toolbar button / draw a circle to call this: the menu
     *  closes and a big blue pointer arms — stare at the new forward for
     *  2x the gaze delay and the snap fires there. */
    fun requestAim() { aimRequest = true }
    /** Screen-position calibration (§8.6): UI-thread request; the GL thread
     *  runs a SCREEN_SWEEP_MS window in which dwell is suspended and the
     *  reticle shrinks to a point, then recenters the head tracker and the
     *  app basis together. The screenpos menu button that used to fire it
     *  is gone; the entry point stays for future binds. */
    @Volatile private var sweepRequest = false
    private var screenSweep = false
    private var screenSweepT0 = 0L
    fun requestScreenSweep() { sweepRequest = true }
    /** Tap while armed cancels the aim instead of snapping. True if consumed. */
    fun cancelAim(): Boolean {
        if (!aimArmed && !aimRequest) return false
        aimArmed = false; aimRequest = false; aimProg = 0f
        FileLog.i("LimpetVR-menu", "aim cancelled")
        return true
    }

    /** Aim dwell: progress grows while still (2x the menu dwell), drains
     *  on motion; at full the basis snaps to the faced direction. */
    private fun updateAim() {
        val nowMs = now()
        val dtMs = (nowMs - aimT).coerceIn(0L, 500L)
        aimT = nowMs
        if (nowMs - aimArmedAt > 15000L) {
            aimArmed = false; aimProg = 0f
            FileLog.i("LimpetVR-menu", "aim timeout")
            return
        }
        val still = synchronized(headViewM) { menuStill.update(headViewM, nowMs) }
        if (still) aimProg += dtMs / dwellMs.toFloat()
        else aimProg = maxOf(0f, aimProg - dtMs / 600f)
        val d = panelDistM
        val f = lastEffFwd
        aimW[0] = f[0] * d; aimW[1] = f[1] * d; aimW[2] = f[2] * d
        if (aimProg >= 1f) {
            aimProg = 0f; aimArmed = false
            recenter("aim")
            FileLog.i("LimpetVR-menu", "aim FIRE -> recenter")
        }
    }

    /** Circle-to-recenter: signed turning angle of the forward-vector
     *  trail in the x/y plane. ONE full loop accumulates to ±~300°;
     *  nods, shakes and look-and-returns self-cancel to ~0, which is
     *  what makes circles robust where linear swipes false-positive.
     *  Fires aim mode (never a blind snap). */
    private fun updateCircle() {
        val nowMs = now()
        if (nowMs < circCooldownUntil) return
        if (nowMs - circLastT < 33L) return
        circLastT = nowMs
        val f = lastEffFwd
        circT[circHead] = nowMs; circX[circHead] = f[0]; circY[circHead] = f[1]
        circHead = (circHead + 1) % circT.size
        if (circN < circT.size) circN++
        // window: trailing 5000ms, oldest -> newest
        var m = 0
        var mx = 0f; var my = 0f
        var idx = (circHead - circN + circT.size * 2) % circT.size
        for (k in 0 until circN) {
            val t = circT[idx]
            if (nowMs - t <= 5000L) {
                winT[m] = t; winX[m] = circX[idx]; winY[m] = circY[idx]
                mx += winX[m]; my += winY[m]; m++
            }
            idx = (idx + 1) % circT.size
        }
        if (m < 18) return
        val span = winT[m - 1] - winT[0]
        if (span < 600L) return
        mx /= m; my /= m
        var maxR = 0f
        for (i in 0 until m) {
            val dx = winX[i] - mx; val dy = winY[i] - my
            maxR = maxOf(maxR, kotlin.math.sqrt(dx * dx + dy * dy))
        }
        val ex = winX[m - 1] - winX[0]; val ey = winY[m - 1] - winY[0]
        val closure = kotlin.math.sqrt(ex * ex + ey * ey)
        var accum = 0.0; var pos = 0; var neg = 0; var travel = 0f
        for (i in 1 until m - 1) {
            val ax = winX[i] - winX[i - 1]; val ay = winY[i] - winY[i - 1]
            val bx = winX[i + 1] - winX[i]; val by = winY[i + 1] - winY[i]
            val la = kotlin.math.sqrt(ax * ax + ay * ay)
            val lb = kotlin.math.sqrt(bx * bx + by * by)
            if (la < 0.004f || lb < 0.004f) continue
            travel += la
            val a = Math.atan2((ax * by - ay * bx).toDouble(), (ax * bx + ay * by).toDouble())
            accum += a
            if (a > 0) pos++ else neg++
        }
        val signFrac = if (pos + neg > 0) maxOf(pos, neg).toFloat() / (pos + neg) else 0f
        val accDeg = Math.toDegrees(accum)
        fun stats() = "span=${span}ms travel=${"%.2f".format(travel)} " +
            "acc=${accDeg.toInt()}° maxR=${"%.3f".format(maxR)} " +
            "close=${"%.3f".format(closure)} sign=${"%.2f".format(signFrac)}"
        // sign ≥0.60: measured real circles score 0.65-0.74 (heads
        // wobble), random motion ~0.52 — margin on both sides
        if (maxR in 0.09f..0.65f && closure <= 0.18f && travel >= 1.0f &&
            kotlin.math.abs(accum) >= 5.2 && signFrac >= 0.6f
        ) {
            circN = 0
            circCooldownUntil = nowMs + 3000L
            aimRequest = true
            FileLog.i("LimpetVR-circle", "FIRE x2 ${stats()}")
            return
        }
        // tuning capture: real rotation that didn't qualify (throttled 2s).
        // The numbers say which guard failed: acc (need ±300° one loop),
        // maxR (need 0.09..0.65 ≈ 5..40° radius), close (need ≤0.18),
        // travel (need ≥1.0), sign (need ≥0.60), span.
        if (kotlin.math.abs(accum) > 2.6 && nowMs - circNearT > 2000L) {
            circNearT = nowMs
            FileLog.i("LimpetVR-circle", "near-miss ${stats()}")
        }
    }
    // scratch window for the circle detector (fields, never allocated per frame)
    private val winT = LongArray(160)
    private val winX = FloatArray(160)
    private val winY = FloatArray(160)

    private fun updateMenu() {
        // recenter aim flow: consume the UI-thread request, close the menu,
        // arm the big blue pointer (suppresses the open logic below)
        if (aimRequest) {
            aimRequest = false
            if (menuWasOpen) FileLog.i("LimpetVR-menu", "menu close (aim)")
            menuOpen = false; menuHitValid = false
            menuHighlight = -2; menuDwellFiredFor = -3
            menuSeekHoverU = -1f
            menuBelowSince = 0L
            menuWasOpen = false
            menuProgFresh = true
            aimArmed = true; aimProg = 0f; aimT = now(); aimArmedAt = aimT
            circN = 0
            FileLog.i("LimpetVR-menu", "aim armed")
        }
        if (aimArmed) { updateAim(); return }
        // circle gesture: menu-closed VIDEO only; stale arcs die when the menu opens
        if (!menuOpen) { if (circleGestureEnabled) updateCircle() }
        else if (circN > 0) circN = 0
        // Trigger metric: head-TILT (angle of the head-up vector from
        // vertical), NOT gaze elevation. asin(fwd.y) conflates yaw with
        // pitch once the head is tilted back: yawing ±30° about the tilted
        // neck axis swings gaze on a cone whose elevation falls ~20°
        // (77°→57°), closing the menu exactly when reaching for the end
        // buttons. Head-up is preserved by yaw (local-Y rotation) exactly,
        // so tilt = atan2(up.z, up.y) is yaw-invariant: + = tipped back,
        // - = tipped forward, roll reads ~0 (never opens).
        val up = lastEffUp
        val tilt = Math.toDegrees(kotlin.math.atan2(up[2].toDouble(), up[1].toDouble())).toFloat()
        val ang = menuOpenAngle()
        val below = menuIsBelow()
        // Open exactly at the angle; close a small band below it (see
        // closeAt). Closing additionally needs 400ms continuously below,
        // killing sensor-noise flapping at the boundary.
        val openAt = ang
        // Close below the open angle (not at the panel edge): the panel
        // stays put while the gaze wanders around and below it, and only
        // hides once the gaze drops past the band and stays there 400ms.
        // The band is deliberately small (1°) but nonzero — tilt dips
        // ~15-20° while operating the end buttons, so closing at the panel
        // edge strobes the menu mid-dwell and forces a re-dip + re-aim to
        // bring it back. Opening is still exact at the angle.
        val closeAt = (ang - 1f).coerceAtLeast(2.5f)
        val above = if (!below) tilt >= (if (!menuOpen) openAt else closeAt)
            else tilt <= -(if (!menuOpen) openAt else closeAt)
        val isUp: Boolean
        if (above) {
            menuBelowSince = 0L
            isUp = true
        } else if (!menuOpen) {
            menuBelowSince = 0L
            isUp = false
        } else {
            if (menuBelowSince == 0L) menuBelowSince = now()
            isUp = now() - menuBelowSince < 400
        }
        if (!isUp) {
            menuOpen = false; menuHitValid = false
            menuHighlight = -2; menuDwellFiredFor = -3
            menuSeekHoverU = -1f
            menuBelowSince = 0L
            if (menuWasOpen) FileLog.i("LimpetVR-menu", "menu close")
            menuWasOpen = false
            menuProgFresh = true
            return
        }
        menuOpen = true
        menuWasOpen = true
        // fresh progress each open: stale banks must never insta-fire
        if (menuProgFresh) {
            menuProgFresh = false
            for (i in menuProg.indices) menuProg[i] = 0f
            menuProgT = now()
        }
        // Windowed stillness: displacement over 250ms, immune to the
        // per-frame sensor jitter that trips instant gates ~30×/s.
        val nowMs = now()
        val still = synchronized(headViewM) { menuStill.update(headViewM, nowMs) }
        // World-locked panel (no yaw following): the head-forward ray meets
        // its plane and the button table says what was hit. Uses the
        // animated elevation so the pointer tracks the ⇅ flip.
        if (menuHitTest()) {
            menuHitValid = true
            val id = menuHitId
            menuSeekHoverU = if (id == -1) seekFrac(menuHitU) else -1f
            updateTooltip(id)
            // Leaky dwell: adopt immediately; progress grows while still
            // on target and drains slowly otherwise. Churn and motion
            // only dent progress instead of zeroing the timer.
            if (id != menuHighlight) {
                menuHighlight = id; menuDwellFiredFor = -3
                // cap carried progress on adopt: churn can never bank
                // a full dwell, and stale slots can't insta-fire
                if (menuUsable(id)) menuProg[menuSlot(id)] = minOf(menuProg[menuSlot(id)], 0.3f)
                FileLog.i("LimpetVR-menu", "dwell start: id=$id tilt=${tilt.toInt()}°")
            }
            val slot = menuSlot(id)
            val dtMs = (nowMs - menuProgT).coerceIn(0L, 500L)
            menuProgT = nowMs
            for (i in menuProg.indices)
                if (i != slot) menuProg[i] = maxOf(0f, menuProg[i] - dtMs / 600f)
            // -2 = decorative (title/backdrop): hover shows the pointer,
            // never fires. Hold shrunken state after firing; re-entry
            // restarts it.
            if (still && menuUsable(id) && menuHighlight != menuDwellFiredFor)
                menuProg[slot] += dtMs / dwellMs.toFloat()
            if (still && menuUsable(id) && menuProg[slot] >= 1f && menuHighlight != menuDwellFiredFor) {
                menuDwellFiredFor = menuHighlight
                menuProg[slot] = 1f
                FileLog.i("LimpetVR-menu", "FIRE id=$id")
                if (id == -1) onMenuEvent(MenuEvent.Seek(seekFrac(menuHitU)))
                else onMenuEvent(MenuEvent.Press(id))
                return
            }
            return
        }
        // off-panel: log the transition once (menuHitValid still true from
        // the last hitting frame), then clear highlight (progress per slot
        // is kept and drains slowly, so brief leaves are forgiven)
        if (menuHitValid)
            FileLog.i("LimpetVR-menu", "left panel (was id=$menuHighlight)")
        menuHitValid = false
        menuSeekHoverU = -1f
        tooltipVisible = false
        if (menuHighlight != -2) {
            menuHighlight = -2; menuDwellFiredFor = -3
        }
    }

    /** Where the head-forward ray meets the play-menu plane, in DESIGN
     *  panel space (the frame the bitmap and the button table are authored
     *  in). False when the ray misses the panel entirely. */
    private fun menuHitTest(): Boolean {
        val o = invHeadWorldM
        val d = lastEffFwd
        val el = Math.toRadians(menuElevCurrent().toDouble()).toFloat()
        val ce = kotlin.math.cos(el); val se = kotlin.math.sin(el)
        // plane through C = (0, se·R, −ce·R) with normal n = (0,−se, ce):
        // y·se − z·ce = R.  Solve o + t·d against it.
        val den = d[2] * ce - d[1] * se
        if (kotlin.math.abs(den) < 1e-6f) return false
        val t = (o[13] * se - o[14] * ce - panelDistM) / den
        if (t <= 0f) return false
        val px = o[12] + t * d[0]
        val py = o[13] + t * d[1]
        val pz = o[14] + t * d[2]
        val k = menuScale()
        val u = px / k
        val v = (py * ce + pz * se) / k
        if (kotlin.math.abs(u) > MENU_RECT_HW || v < MENU_Y0 || v > MENU_Y1) return false
        menuHitU = u; menuHitV = v
        // buttons first: they sit inside the backdrop and must win.
        var id = -2
        for (b in menuButtons)
            if (kotlin.math.abs(u - b.x) <= b.hw && kotlin.math.abs(v - b.y) <= b.hh) { id = b.id; break }
        if (id == -2 && kotlin.math.abs(u - menuBar.x) <= menuBar.hw &&
            kotlin.math.abs(v - menuBar.y) <= menuBar.hh
        ) id = -1
        menuHitId = id
        return true
    }

    /** Tooltip pill text for whatever the gaze is on: the button's name, or
     *  the time a seek-bar hover would jump to. */
    private fun updateTooltip(id: Int) {
        val txt = when {
            !enableTooltip -> ""
            id == -1 && menuDurMs > 0 && menuSeekHoverU >= 0f ->
                fmtTime((menuSeekHoverU.coerceIn(0f, 1f) * menuDurMs).toLong())
            id in MENU_NAMES.indices -> MENU_NAMES[id]
            else -> ""
        }
        tooltipText = txt
        tooltipVisible = txt.isNotEmpty()
    }

    // ---------- drawing ----------
    private fun drawVideo(eye: Int) {
        val m = mesh ?: return
        buildVideoModel()
        // Panoramic video rides the FOV-scaled projection; the FLAT screen
        // shares the unscaled one with every panel (§6.3).
        val projBase = if (effProj() == Projection.FLAT) ovM else domeOvM
        Matrix.multiplyMM(mvpM, 0, projBase, 0, modelM, 0)
        if (effProj() == Projection.FISHEYE) {
            drawVideoFisheye(eye, m)
            return
        }
        GLES20.glUseProgram(progOes)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureId)
        GLES20.glUniform1i(uTexOes, 0)
        GLES20.glUniform1i(uStereoOes, when (stereo) { Stereo.MONO -> 0; Stereo.SBS -> 1; Stereo.TB -> 2 })
        GLES20.glUniform1i(uEyeOes, eye)
        GLES20.glUniformMatrix4fv(uTexMatOes, 1, false, texMat, 0)
        // Texture zoom lives only on the finite FLAT quad (§5); domes keep
        // the full sampled range here and slide the model instead.
        GLES20.glUniform1f(uZoomOutOes, if (effProj() == Projection.FLAT) flatZoomF() else 1f)
        GLES20.glUniformMatrix4fv(uMvpOes, 1, false, mvpM, 0)
        GLES20.glEnableVertexAttribArray(aPosOes)
        GLES20.glVertexAttribPointer(aPosOes, 3, GLES20.GL_FLOAT, false, 0, m.verts)
        GLES20.glEnableVertexAttribArray(aTexOes)
        GLES20.glVertexAttribPointer(aTexOes, 2, GLES20.GL_FLOAT, false, 0, m.tex)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, m.indexCount, GLES20.GL_UNSIGNED_SHORT, m.indices)
        GLES20.glDisableVertexAttribArray(aPosOes)
        GLES20.glDisableVertexAttribArray(aTexOes)
    }

    // ---------- web mode ----------
    private val IDENTITY16 = FloatArray(16).also { Matrix.setIdentityM(it, 0) }

    /** The page on the flat screen: same mesh, model, curve and zoom range
     *  as the 2D video, but a plain 2D texture (the page is a capture, not
     *  a decoder surface), so it needs its own tiny program for the zoom. */
    private fun drawWeb(eye: Int) {
        val m = mesh ?: return
        if (webTexId < 0 || webConsumedFrames == 0L) return
        buildVideoModel()
        Matrix.multiplyMM(mvpM, 0, ovM, 0, modelM, 0)
        GLES20.glUseProgram(progWeb)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, webTexId)
        GLES20.glUniform1i(uTexWeb, 0)
        GLES20.glUniform1f(uZoomWeb, webZoomF())
        GLES20.glUniformMatrix4fv(uMvpWeb, 1, false, mvpM, 0)
        GLES20.glEnableVertexAttribArray(aPosWeb)
        GLES20.glVertexAttribPointer(aPosWeb, 3, GLES20.GL_FLOAT, false, 0, m.verts)
        GLES20.glEnableVertexAttribArray(aTexWeb)
        GLES20.glVertexAttribPointer(aTexWeb, 2, GLES20.GL_FLOAT, false, 0, m.tex)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, m.indexCount, GLES20.GL_UNSIGNED_SHORT, m.indices)
        GLES20.glDisableVertexAttribArray(aPosWeb)
        GLES20.glDisableVertexAttribArray(aTexWeb)
    }

    /** World point of page texture coord (u,v) — the exact geometry the
     *  flat screen mesh uses (plane at curve 0, eye-centred cap blend above
     *  it), including buildVideoModel's +0.25 lift. v is GL texture space
     *  (v=1 is the TOP row of the page). */
    private fun webPointAt(u: Float, v: Float, out: FloatArray) {
        val (w, h) = screenDims()
        val c = screenCurve.coerceIn(0f, 1f)
        val fx = (u - 0.5f) * w
        val fy = (v - 0.5f) * h
        if (c <= 0f) {
            out[0] = fx; out[1] = fy + 0.25f; out[2] = -FLAT_DIST
            return
        }
        val arcMax = Math.toRadians(SCREEN_ARC_MAX_DEG.toDouble()).toFloat()
        val rt = maxOf(SCREEN_R_MIN, w / arcMax)
        val ax = (u - 0.5f) * w / rt
        val by = (v - 0.5f) * h / rt
        val cyb = kotlin.math.cos(by); val syb = kotlin.math.sin(by)
        val cx = rt * cyb * kotlin.math.sin(ax)
        val ry = rt * syb
        val cz = -rt * cyb * kotlin.math.cos(ax)
        out[0] = fx + (cx - fx) * c
        out[1] = fy + (ry - fy) * c + 0.25f
        out[2] = -FLAT_DIST + (cz + FLAT_DIST) * c
    }

    /** Gaze ray -> page texture coord by solving P(u,v) parallel to the ray
     *  with Newton iterations (the curved cap is not a plane, so a single
     *  plane intersection would miss by most of a button at curve 1).
     *  Null when the ray leaves the screen. */
    @Volatile private var webGazeDist = -1f

    private fun webGazeUv(fwd: FloatArray): FloatArray? {
        val hx = invHeadWorldM[12]; val hy = invHeadWorldM[13]; val hz = invHeadWorldM[14]
        val p = FloatArray(3); val pu = FloatArray(3); val pv = FloatArray(3)
        var u = 0.5f; var v = 0.5f
        val h = 0.002f
        for (it in 0 until 5) {
            webPointAt(u, v, p)
            val qx = p[0] - hx; val qy = p[1] - hy; val qz = p[2] - hz
            val fx = qy * fwd[2] - qz * fwd[1]
            val fy = qz * fwd[0] - qx * fwd[2]
            if (kotlin.math.abs(fx) < 2e-4f && kotlin.math.abs(fy) < 2e-4f) break
            webPointAt(u + h, v, pu); webPointAt(u, v + h, pv)
            val a = pu[0] - hx; val b = pu[1] - hy; val cc = pu[2] - hz
            val d0 = pv[0] - hx; val d1 = pv[1] - hy; val d2 = pv[2] - hz
            val jx1 = b * fwd[2] - cc * fwd[1]; val jy1 = cc * fwd[0] - a * fwd[2]
            val jx2 = d1 * fwd[2] - d2 * fwd[1]; val jy2 = d2 * fwd[0] - d0 * fwd[2]
            val a11 = (jx1 - fx) / h; val a12 = (jx2 - fx) / h
            val a21 = (jy1 - fy) / h; val a22 = (jy2 - fy) / h
            val det = a11 * a22 - a12 * a21
            if (kotlin.math.abs(det) < 1e-9f) return null
            u = (u + (-fx * a22 + a12 * fy) / det).coerceIn(-0.4f, 1.4f)
            v = (v + (-a11 * fy + a21 * fx) / det).coerceIn(-0.4f, 1.4f)
        }
        // Off the page the solution still lies on the screen PLANE, so return
        // it (outside 0..1) rather than null: that gives the reticle a target
        // which keeps moving smoothly as the gaze leaves the page, instead of
        // pinning it to the page edge where it stalled in the gap. Only a ray
        // that stops facing the plane returns null.
        if (u > -0.4f && u < 1.4f && v > -0.4f && v < 1.4f) {
            webPointAt(u, v, p)
            val dx = p[0] - hx; val dy = p[1] - hy; val dz = p[2] - hz
            webGazeDist = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
            return floatArrayOf(u, v)
        }
        return null
    }

    /** Scrollbar strip geometry, in screen texture space. Fixed, so the
     *  mesh only rebuilds when the screen (curve/size) changes. The strip is
     *  deliberately wide and the arrows tall: they are the only gaze targets
     *  on the page, and a few pixels of head wobble used to mean "miss".
     *
     *  u > 1 puts the strip just OUTSIDE the right edge of the screen. It used
     *  to sit inside the page (0.930-0.994), which was invisible while the
     *  page was zoomed in and smaller than the screen, but now that the page
     *  always fills the screen exactly, an inside strip lies on top of the
     *  page's right-hand content. Outside, it stays adjacent to the page
     *  edge with nothing covered. webPointAt extrapolates linearly past the
     *  screen (and round the cap when curved), so u > 1 is a real position
     *  just beyond the surface, not a clamp. */
    private val barU0 = 1.010f
    private val barU1 = 1.074f
    private val barV0 = 0.010f
    private val barV1 = 0.990f
    private val BAR_ARROW = 0.090f
    /** Gaze slop around a bar target, in screen units. */
    private val BAR_SLOP_U = 0.020f
    private val BAR_SLOP_V = 0.030f

    /** -1 / 1 when the gaze is on a scrollbar arrow, else 0. The slop makes
     *  the small targets forgiving of a pixel or two of mapping error.
     *  v is screen texture space: v = 1 is the TOP of the screen. */
    private fun webBarDir(u: Float, v: Float): Int {
        if (!webScrollable) return 0
        if (u < barU0 - BAR_SLOP_U || u > barU1 + BAR_SLOP_U) return 0
        if (v < barV0 - BAR_SLOP_V || v > barV1 + BAR_SLOP_V) return 0
        if (v >= barV1 - BAR_ARROW) return -1   // top arrow: page up
        if (v <= barV0 + BAR_ARROW) return 1    // bottom arrow: page down
        return 0
    }

    /** Dwell on the page: arrows scroll, links/controls click, inert page
     *  opens the web panel. Same leaky dwell + stillness gate as the menu. */
    private fun updateWebGaze() {
        val nowMs = now()
        val dtMs = (nowMs - webProgT).coerceIn(0L, 500L)
        webProgT = nowMs
        val uv = synchronized(headViewM) { webGazeUv(lastEffFwd) }
        val still = synchronized(headViewM) { webStill.update(headViewM, nowMs) }
        if (uv == null) { webPagePointOk = false; webPageDist = -1f; webProgF = maxOf(0f, webProgF - dtMs / 600f); return }

        val dir = webBarDir(uv[0], uv[1])
        // The page is drawn with no texture zoom, so the texture coord under
        // the reticle IS the screen coord. The inversion that used to live
        // here was the source of a run of gaze/page misalignments: it fed
        // the reticle the unzoomed texel position, which is only the same
        // point while zoom happens to be 1.
        val px = uv[0] * webPageW
        val py = (1f - uv[1]) * webPageH
        noteWebPagePoint(uv[0], uv[1])

        // The dwell target is the STATE under the reticle, not the exact
        // pixel: keying on pixel coordinates made every wobble of the head
        // (VR jitter is never quite still) reset the dwell, so a link could
        // never be stared at long enough to fire. The click still uses the
        // live pixel, so it lands where you are actually looking.
        val target = when {
            dir != 0 -> "bar$dir"
            webHitInteractive -> "hit"
            else -> ""
        }
        // A hit answer is up to ~100 ms stale. Only distrust it if the gaze
        // has genuinely gone somewhere else — a tight radius made the
        // reticle flicker between "on a link" and "not" with every wobble,
        // which reset the dwell forever and nothing ever fired.
        if (webAskPx >= 0f &&
            kotlin.math.hypot(px - webAskPx, py - webAskPy) > WEB_HIT_SLOP
        ) webHitInteractive = false

        if (target != webTarget) {
            webTarget = target
            webFiredFor = ""
            // Blank page (or moved off a control) resets hard: the reticle
            // must not stay shrunk with nothing to activate.
            webProgF = if (target.isEmpty()) 0f else minOf(webProgF, 0.25f)
        }
        // Ask the page what is under the reticle (the answer lands in
        // onWebHit). Throttled, and re-asked after firing so moving away and
        // back is a fresh act.
        // Keep the hit state live even after a fire: gating the refresh on
        // the latch deadlocked it (the target can only change if we keep
        // asking, so the first click froze dwell for the whole page).
        if (dir == 0 && nowMs - webHitAskedAt > 100L) {
            webHitAskedAt = nowMs
            webHitToken++
            webAskPx = px; webAskPy = py
            onWebEvent(WebEvent.HitTest(webHitToken, px, py))
        }
        // Nothing to activate here: wait (the panel opens on gaze angle, not
        // by staring at blank page).
        if (target.isEmpty()) { webProgF = 0f; return }

        if (webFiredFor == webTarget) webProgF = 1f   // held at full shrink until the gaze leaves
        else if (still) webProgF += dtMs / dwellMs.toFloat()
        else webProgF = maxOf(0f, webProgF - dtMs / 600f)
        if (still && webProgF >= 1f && webFiredFor != webTarget) {
            webFiredFor = webTarget
            webProgF = 1f
            FileLog.i("LimpetVR-web", "FIRE target=$webTarget px=$px py=$py")
            when {
                dir != 0 -> onWebEvent(WebEvent.Scroll(dir))
                else -> onWebEvent(WebEvent.Click(px, py))
            }
            // Re-arm once the gaze leaves, so a second stare is a second act.
        }
    }

    private fun webDwellProg(): Float = webProgF.coerceIn(0f, 1f)

    /** TEMP DEBUG: where the gaze lands on the page, tracked every frame
     *  INCLUDING while the web panel holds the gaze (otherwise the on-page
     *  crosshair froze the moment you looked at the panel). */
    /** The page pixel currently under the gaze, in world space. The reticle
     *  is drawn AT this point, not at a distance along the ray, so it lands
     *  on the same surface and converges with the page in both eyes. */
    @Volatile private var webPagePoint = FloatArray(3)
    @Volatile private var webPagePointOk = false
    @Volatile private var webPageDist = -1f

    /** World point on the ACTUAL page mesh at texture coord (u,v).
     *
     *  The mesh is a regular row-major grid, so the cell is found directly
     *  from (u,v) instead of by hunting for a near vertex. The earlier
     *  nearest-vertex search stepped the v-neighbour by ONE index, but in a
     *  row-major grid the v-neighbour is a whole row away - so it
     *  interpolated against another u-neighbour and the result snapped to
     *  grid rows, which showed up as chunky vertical reticle motion while
     *  horizontal stayed smooth.
     *
     *  Reading the real vertices (rather than the analytic webPointAt model)
     *  means this cannot disagree with the surface actually drawn.
     */
    private fun webPointOnMesh(u: Float, v: Float, out: FloatArray): Boolean {
        val m = mesh ?: return false
        val vs = m.tex; val ps = m.verts
        val nv = vs.capacity() / 2
        if (nv < 4 || ps.capacity() < nv * 3) return false
        fun tu(i: Int) = vs.get(i * 2)
        fun tv(i: Int) = vs.get(i * 2 + 1)
        fun pu(i: Int, k: Int) = ps.get(i * 3 + k)

        // Row length: u climbs across a row and wraps at the row end.
        var rowLen = 1
        while (rowLen < nv && tu(rowLen) > tu(rowLen - 1)) rowLen++
        if (rowLen < 2 || nv % rowLen != 0) return false
        val rows = nv / rowLen
        if (rows < 2) return false

        // Which row does texture v belong to? Measured on the real mesh:
        // row 0 carries v = 1 and the last row v = 0, so v = 1 is the FIRST
        // row. Guessing this sign wrong mirrors the lookup vertically - and a
        // crosshair cannot reveal it, being symmetric top-to-bottom.
        val firstRowV = tv(0)
        val lastRowV = tv((rows - 1) * rowLen)
        val vIsRowIndex = firstRowV <= lastRowV   // v climbs down the rows

        val fx = (u.coerceIn(0f, 1f)) * (rowLen - 1)
        val fyRaw = (v.coerceIn(0f, 1f)) * (rows - 1)
        val fy = if (vIsRowIndex) fyRaw else (rows - 1) - fyRaw

        var cx = kotlin.math.floor(fx.toDouble()).toInt()
        var cy = kotlin.math.floor(fy.toDouble()).toInt()
        if (cx > rowLen - 2) cx = rowLen - 2
        if (cx < 0) cx = 0
        if (cy > rows - 2) cy = rows - 2
        if (cy < 0) cy = 0
        val ax = fx - cx
        val ay = fy - cy

        val i00 = cy * rowLen + cx
        val i10 = i00 + 1
        val i01 = i00 + rowLen
        val i11 = i01 + 1
        for (k in 0..2) {
            val p00 = pu(i00, k); val p10 = pu(i10, k)
            val p01 = pu(i01, k); val p11 = pu(i11, k)
            val top = p00 + (p10 - p00) * ax
            val bot = p01 + (p11 - p01) * ax
            out[k] = top + (bot - top) * ay
        }
        // The mesh is drawn through modelM (the +0.25 lift, and the flat
        // distance or cap z), so apply it here too or the point lands in the
        // wrong space entirely.
        buildVideoModel()
        val wx = out[0]; val wy = out[1]; val wz = out[2]
        out[0] = modelM[0] * wx + modelM[4] * wy + modelM[8] * wz + modelM[12]
        out[1] = modelM[1] * wx + modelM[5] * wy + modelM[9] * wz + modelM[13]
        out[2] = modelM[2] * wx + modelM[6] * wy + modelM[10] * wz + modelM[14]
        return true
    }

    /** Gaze ray -> the web panel's surface. Writes the world point into
     *  [out] and returns true ONLY when the ray genuinely lands inside the
     *  panel rectangle.
     *
     *  Deliberately no clamping. Clamping the plane intersection to the rect
     *  pinned the reticle to the panel edge whenever the gaze left the panel,
     *  so it latched there and would not come back down until a recenter; and
     *  a ray passing below the panel still hits the same infinite plane, just
     *  on the far side, which flipped the reticle from below the page to
     *  above it. Rejecting off-panel hits instead lets the reticle fall back
     *  to the page, which is the surface actually being looked at. */
    private fun webPanelPoint(out: FloatArray): Boolean {
        val o = invHeadWorldM
        val fwd = lastEffFwd
        val d = panelDistM
        val el = Math.toRadians(browserElevDeg.toDouble()).toFloat()
        val cy = kotlin.math.sin(el) * d
        val cz = -kotlin.math.cos(el) * d
        var pny = -cy; var pnz = -cz
        val pnl = kotlin.math.sqrt(pny * pny + pnz * pnz).coerceAtLeast(1e-6f)
        pny /= pnl; pnz /= pnl
        val denom = fwd[1] * pny + fwd[2] * pnz
        if (denom >= -0.05f) return false      // edge-on or facing away
        val t = ((cy - o[13]) * pny + (cz - o[14]) * pnz) / denom
        if (t <= 0f) return false
        val hx = o[12] + fwd[0] * t
        val hy = o[13] + fwd[1] * t
        val hz = o[14] + fwd[2] * t
        val upx = 0f; val upy = pnz; val upz = -pny
        val upl = kotlin.math.sqrt(upy * upy + upz * upz).coerceAtLeast(1e-6f)
        val hw = panelHalfW(); val hh = panelHalfH()
        val alongRight = hx
        val alongUp = (hy - cy) * (upy / upl) + (hz - cz) * (upz / upl)
        if (alongRight < -hw || alongRight > hw) return false
        if (alongUp < -hh || alongUp > hh) return false
        out[0] = hx; out[1] = hy; out[2] = hz
        return true
    }

    /** Where the reticle was last drawn, and the eased position now being
     *  drawn. The page and the panel are at very different depths (12 m vs
     *  ~4 m), so switching surface snapped the reticle; easing over ~120 ms
     *  turns that snap into a short glide without adding lag to normal
     *  tracking, since the target itself is still followed every frame. */
    private val webLastPoint = FloatArray(3)
    @Volatile private var webLastPointOk = false
    private val webSmoothPoint = FloatArray(3)
    @Volatile private var webSmoothOk = false
    private var webSmoothT = 0L

    /** Places the reticle on the gaze RAY at [dist] from the head, seeded
     *  from the current smoothed point so crossing into or out of the gap is
     *  continuous. Used only where neither surface is under the gaze. */
    private fun webReticleRay(dist: Float) {
        val o = invHeadWorldM
        val f = lastEffFwd
        if (!webSmoothOk) {
            for (i in 0..2) webSmoothPoint[i] = o[12 + i] + f[i] * dist
            webSmoothOk = true
        }
        val nowMs = now()
        val dt = ((nowMs - webSmoothT).coerceIn(0L, 100L)).toFloat() / 100f
        webSmoothT = nowMs
        val k = 1f - kotlin.math.exp(-dt / 0.12f)
        for (i in 0..2) {
            val t = o[12 + i] + f[i] * dist
            webSmoothPoint[i] += (t - webSmoothPoint[i]) * k
        }
        webLastPoint[0] = webSmoothPoint[0]
        webLastPoint[1] = webSmoothPoint[1]
        webLastPoint[2] = webSmoothPoint[2]
        webLastPointOk = true
        Matrix.setIdentityM(tmpA, 0)
        Matrix.translateM(tmpA, 0, webSmoothPoint[0], webSmoothPoint[1], webSmoothPoint[2])
    }

    /** Half-size of the reticle quad, from the distance to where it is
     *  ACTUALLY drawn. Sizing from a nominal distance (the page depth or the
     *  panel depth, whichever branch we took) is what made the reticle
     *  shrink to about half when it was drawn on the far page using the
     *  panel's near distance, and why a recenter appeared to "fix" it. */
    private fun webRetS(sizeMul: Float, prog: Float): Float {
        val hx = invHeadWorldM[12]; val hy = invHeadWorldM[13]; val hz = invHeadWorldM[14]
        val dx = webSmoothPoint[0] - hx
        val dy = webSmoothPoint[1] - hy
        val dz = webSmoothPoint[2] - hz
        val dist = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz).coerceAtLeast(0.5f)
        return kotlin.math.sin(RETICLE_ANG) * dist * sizeMul * (1f - 0.85f * prog.coerceIn(0f, 1f))
    }

    /** Builds tmpA for the reticle quad at [target], easing from the
     *  previous position. Callers premultiply tmpA by ovM. */
    private fun webReticleAt(target: FloatArray) {
        val nowMs = now()
        val dt = ((nowMs - webSmoothT).coerceIn(0L, 100L)).toFloat() / 100f
        webSmoothT = nowMs
        if (!webLastPointOk) {
            System.arraycopy(target, 0, webLastPoint, 0, 3)
            System.arraycopy(target, 0, webSmoothPoint, 0, 3)
            webLastPointOk = true; webSmoothOk = true
        } else {
            System.arraycopy(target, 0, webLastPoint, 0, 3)
            if (!webSmoothOk) {
                System.arraycopy(target, 0, webSmoothPoint, 0, 3)
                webSmoothOk = true
            } else {
                // Frame-rate independent exponential approach.
                val k = 1f - kotlin.math.exp(-dt / 0.12f)
                for (i in 0..2) webSmoothPoint[i] += (target[i] - webSmoothPoint[i]) * k
            }
        }
        Matrix.setIdentityM(tmpA, 0)
        Matrix.translateM(tmpA, 0, webSmoothPoint[0], webSmoothPoint[1], webSmoothPoint[2])
    }

    private fun noteWebPagePoint(u: Float, v: Float) {
        // Off the page the mesh lookup clamps to the page edge, which parked
        // the reticle there and made it pause in the gap above the page. The
        // analytic form extends smoothly past the edge instead, so the target
        // keeps moving and the reticle glides up into the panel.
        if (u >= 0f && u <= 1f && v >= 0f && v <= 1f)
            webPointOnMesh(u, v, webPagePoint)
        else
            webPointAt(u, v, webPagePoint)
        val hx = invHeadWorldM[12]; val hy = invHeadWorldM[13]; val hz = invHeadWorldM[14]
        val dx = webPagePoint[0] - hx
        val dy = webPagePoint[1] - hy
        val dz = webPagePoint[2] - hz
        webPageDist = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
        webPagePointOk = true
    }

    /** Track where the gaze lands on the page plane. Runs whether or not the
     *  panel is open, and whether or not the debug crosshair is on, because
     *  the reticle depends on it. */
    fun updateWebPagePoint() {
        if (mode != Mode.WEB) return
        val uv = synchronized(headViewM) { webGazeUv(lastEffFwd) } ?: run {
            webPagePointOk = false
            return
        }
        noteWebPagePoint(uv[0], uv[1])
    }

    fun updateWebDebugPoint() {
        if (!webDbgOn || mode != Mode.WEB) return
        val uv = synchronized(headViewM) { webGazeUv(lastEffFwd) } ?: return
        webDebugPoint = floatArrayOf(uv[0] * webPageW, (1f - uv[1]) * webPageH)
    }

    /** Keep the web panel shut for a moment. The gaze is often still over it
     *  right after it was used (a bookmark row, a recenter), and letting it
     *  spring open again takes the gaze off the page it just loaded. */
    fun webLockPanel() { webPanelHold = 0f; webPanelLockUntil = now() + (PANEL_TOGGLE_MS * 6).toLong() }

    /** Forget the fired latch, so a freshly loaded page's links are dwellable
     *  straight away. */
    fun webResetDwell() {
        webFiredFor = "\u0000none"
        webTarget = ""
        webProgF = 0f
        webHitInteractive = false
        webPanelHold = 0f
    }

    /** The web panel opens when the gaze actually goes UP ONTO it, and
     *  closes when it goes back to the page — not on a head-tilt threshold.
     *  A tilt gate was unusable: holding the phone leaves you permanently
     *  past the threshold, so the panel stayed open, swallowed the gaze and
     *  the page never got a single dwell. A short dwell on the panel also
     *  stops a glance while turning from firing it. */
    private fun updateWebPanelTilt() {
        // Just recentred: whatever the gaze lands on, the page keeps it for
        // a moment. Without this, recentring while looking at the panel let
        // it reopen at once and swallow the gaze, and the page looked dead.
        if (now() < webPanelLockUntil) { webPanelHold = 0f; return }
        val onPanel = webGazeOnPanel()
        val dt = (now() - webPanelT0).coerceIn(0L, 500L)
        webPanelT0 = now()
        if (onPanel) {
            webPanelHold = minOf(1f, webPanelHold + dt / PANEL_TOGGLE_MS)
            if (webPanelHold >= 1f && !webPanelOpen) {
                webPanelOpen = true
                FileLog.i("LimpetVR-web", "web panel open (gaze on panel)")
                onWebEvent(WebEvent.Panel(true))
            }
        } else {
            webPanelHold = maxOf(0f, webPanelHold - dt / PANEL_TOGGLE_MS)
            if (webPanelHold <= 0f && webPanelOpen) {
                webPanelOpen = false
                // Re-arm the dwell for the page. While the panel was open
                // the page gaze did not run, so the fired latch kept the
                // target it had when you looked up; coming back down onto
                // the same link then read as already-fired and would not
                // activate until a recenter cleared it.
                webResetDwell()
                FileLog.i("LimpetVR-web", "web panel close (gaze back on page)")
                onWebEvent(WebEvent.Panel(false))
            }
        }
    }

    private fun webGazeOnPanel(): Boolean {
        val o = invHeadWorldM
        val fwd = lastEffFwd
        val d = panelDistM
        val el = Math.toRadians(browserElevDeg.toDouble()).toFloat()
        val cy = kotlin.math.sin(el) * d
        val cz = -kotlin.math.cos(el) * d
        var pny = -cy; var pnz = -cz
        val pnl = kotlin.math.sqrt(pny * pny + pnz * pnz).coerceAtLeast(1e-6f)
        pny /= pnl; pnz /= pnl
        val denom = fwd[1] * pny + fwd[2] * pnz
        if (denom >= -0.05f) return false
        val t = ((cy - o[13]) * pny + (cz - o[14]) * pnz) / denom
        if (t <= 0f) return false
        val hy = o[13] + fwd[1] * t
        val hz = o[14] + fwd[2] * t
        val upy = pnz; val upz = -pny
        val upl = kotlin.math.sqrt(upy * upy + upz * upz).coerceAtLeast(1e-6f)
        val alongRight = o[12] + fwd[0] * t
        val alongUp = (hy - cy) * (upy / upl) + (hz - cz) * (upz / upl)
        val hw = panelHalfW(); val hh = panelHalfH()
        return alongRight >= -hw && alongRight <= hw && alongUp >= -hh && alongUp <= hh
    }

    /** The scrollbar: up/down arrows plus a thumb, drawn as a strip pinned
     *  to the right edge of the (possibly curved) screen. */
    private fun drawWebBar() {
        if (!webScrollable) return
        val key = screenCurve.toString() + "|" + screenSize.toString() + "|" +
            webPageW + "x" + webPageH
        if (webBarMesh == null || webBarMeshKey != key) {
            webBarMesh = webBarMeshBuild()
            webBarMeshKey = key
        }
        val m = webBarMesh ?: return
        maybeUploadWebBar()
        drawMesh2d(m, webBarTexId, ovM, 1f)
    }

    private val barP = FloatArray(3)
    private val wpTmp = FloatArray(3)
    private fun webBarMeshBuild(): Mesh {
        // Pull the strip a few cm toward the viewer so it never z-fights
        // the page underneath.
        fun corner(u: Float, v: Float): FloatArray {
            webPointAt(u, v, barP)
            val len = kotlin.math.sqrt(barP[0] * barP[0] + barP[1] * barP[1] + barP[2] * barP[2])
                .coerceAtLeast(1e-3f)
            val k = (len - 0.05f) / len
            return floatArrayOf(barP[0] * k, barP[1] * k, barP[2] * k)
        }
        // The strip bitmap's row 0 is its top (the up arrow), so the mesh
        // maps it straight through: no flip.
        return gridQuadP(
            corner(barU0, barV1), corner(barU1, barV1),
            corner(barU0, barV0), corner(barU1, barV0), 1, 1
        )
    }

    private fun maybeUploadWebBar() {
        val W = 64; val H = 1024
        val bmp = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.TRANSPARENT)
        val p = Paint(Paint.ANTI_ALIAS_FLAG)
        p.color = Color.argb(150, 10, 14, 22)
        c.drawRoundRect(3f, 2f, (W - 3).toFloat(), (H - 2).toFloat(), 14f, 14f, p)
        // arrows
        p.color = Color.argb(245, 235, 240, 248)
        val ay = BAR_ARROW / (barV1 - barV0) * H
        fun tri(cy: Float, up: Boolean) {
            val w = W * 0.34f
            c.drawPath(Path().apply {
                if (up) {
                    moveTo(W / 2f, cy - ay * 0.30f); lineTo(W / 2f - w, cy + ay * 0.28f)
                    lineTo(W / 2f + w, cy + ay * 0.28f)
                } else {
                    moveTo(W / 2f, cy + ay * 0.30f); lineTo(W / 2f - w, cy - ay * 0.28f)
                    lineTo(W / 2f + w, cy - ay * 0.28f)
                }
                close()
            }, p)
        }
        tri(ay * 0.5f, true)
        tri(H - ay * 0.5f, false)
        // thumb: size = visible fraction, position = scroll fraction
        val frac = webScrollFrac.coerceIn(0f, 1f)
        val view = webViewFrac.coerceIn(0.02f, 1f)
        val thumbH = view * (H - 2 * ay)
        val top = ay + (H - 2 * ay - thumbH) * frac
        p.color = Color.argb(215, 125, 211, 252)
        c.drawRoundRect(7f, top, (W - 7).toFloat(), (top + thumbH).toFloat(), 11f, 11f, p)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, webBarTexId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        webBarBitmap?.recycle()
        webBarBitmap = bmp
    }

    /** World size of the flat screen: aspect-corrected width/height for the
     *  plane (§5, base 8.5·screenSize). The shader samples ONE half of an
     *  SBS/TB frame, so the aspect is the per-eye content aspect, not the
     *  full frame. Shared by buildVideoModel() and the curved-cap mesh.
     *  WEB borrows this path with the page's own aspect (and no stereo
     *  split) so the browser sits on exactly the same screen as a video. */
    private fun screenDims(): Pair<Float, Float> {
        val full = if (mode == Mode.WEB) webPageAspect() else videoAspect
        val a = when (if (mode == Mode.WEB) Stereo.MONO else stereo) {
            Stereo.SBS -> full / 2f; Stereo.TB -> full * 2f; else -> full
        }
            .coerceIn(0.25f, 4f)
        val base = 8.5f * screenSize.coerceIn(0.5f, 10f)
        val w = if (a >= 1.7777778f) base else base / 1.7777778f * a
        return w to (w / a)
    }

    /** WEB always rides the FLAT screen (same size/curve/zoom as 2D video),
     *  whatever dome projection the video is set to. */
    private fun effProj(): Projection = if (mode == Mode.WEB) Projection.FLAT else projection

    private fun webPageAspect(): Float =
        (webPageW.toFloat() / webPageH.toFloat().coerceAtLeast(1f)).coerceIn(0.25f, 4f)

    /** Model matrix for the video: FLAT = aspect-correct plane of world
     *  width 8.5·screenSize (unit quad scaled, §5) — or, when screenCurve
     *  > 0, the bent cap whose world size is baked into the mesh, so only
     *  the eye-height lift remains; domes = unit sphere slid toward the
     *  viewer by (zoom − 1). */
    private fun buildVideoModel() {
        Matrix.setIdentityM(modelM, 0)
        if (effProj() == Projection.FLAT) {
            val (w, h) = screenDims()
            if (screenCurve > 0f) {
                // Cap vertices already carry world w/h and their own z.
                Matrix.translateM(modelM, 0, 0f, 0.25f, 0f)
            } else {
                Matrix.translateM(modelM, 0, 0f, 0.25f, -FLAT_DIST)
                Matrix.scaleM(modelM, 0, w / 2f, h / 2f, 1f)
            }
        } else {
            Matrix.translateM(modelM, 0, 0f, 0f, zoom.coerceIn(0.1f, 20f) - 1f)
        }
    }

    /** Fisheye video path (§8): same mesh/basis/projection/zoom as the
     *  equirect path — only the texture lookup differs (equidistant circle
     *  inversion from the per-fragment view direction). */
    private fun drawVideoFisheye(eye: Int, m: Mesh) {
        GLES20.glUseProgram(progFish)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureId)
        GLES20.glUniform1i(uTexFish, 0)
        val st = when (stereo) { Stereo.MONO -> 0; Stereo.SBS -> 1; Stereo.TB -> 2 }
        GLES20.glUniform1i(uStereoFish, st)
        GLES20.glUniform1i(uEyeFish, eye)
        GLES20.glUniformMatrix4fv(uTexMatFish, 1, false, texMat, 0)
        GLES20.glUniform1f(uZoomOutFish, 1f)
        GLES20.glUniformMatrix4fv(uMvpFish, 1, false, mvpM, 0)
        // Active circle: per-layout defaults (§8: half-image center, radius
        // of one quarter frame width) plus calibration offsets.
        val ar = videoAspect.coerceIn(0.5f, 4f)
        val rs = fisheyeRadiusScale.coerceIn(0.25f, 2f)
        val cx: Float; val cy: Float; val ru: Float; val rv: Float
        when (st) {
            1 -> { // side-by-side: left circle in left half, right in right
                cx = eye * 0.5f + 0.25f + fisheyeCxOff
                cy = 0.5f + fisheyeCyOff
                ru = 0.25f * rs; rv = 0.25f * ar * rs
            }
            2 -> { // stacked halves
                cx = 0.5f + fisheyeCxOff
                cy = eye * 0.5f + 0.25f + fisheyeCyOff
                ru = 0.25f * rs; rv = 0.5f * ar * rs
            }
            else -> { // single circle over the full frame
                cx = 0.5f + fisheyeCxOff
                cy = 0.5f + fisheyeCyOff
                ru = 0.5f * rs; rv = 0.5f * ar * rs
            }
        }
        GLES20.glUniform2f(uFishC, cx, cy)
        GLES20.glUniform2f(uFishR, ru, rv)
        GLES20.glUniform1f(uFishMirror, if ((eye == 0 && fisheyeMirrorL) || (eye == 1 && fisheyeMirrorR)) 1f else 0f)
        GLES20.glEnableVertexAttribArray(aPosFish)
        GLES20.glVertexAttribPointer(aPosFish, 3, GLES20.GL_FLOAT, false, 0, m.verts)
        GLES20.glEnableVertexAttribArray(aTexFish)
        GLES20.glVertexAttribPointer(aTexFish, 2, GLES20.GL_FLOAT, false, 0, m.tex)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, m.indexCount, GLES20.GL_UNSIGNED_SHORT, m.indices)
        GLES20.glDisableVertexAttribArray(aPosFish)
        GLES20.glDisableVertexAttribArray(aTexFish)
    }

    /** Bilinear grid over the quad (p00 top-left, p10 top-right, p01
     *  bottom-left, p11 bottom-right). Per-vertex lens warp needs real
     *  vertices across the surface — a 2-triangle quad warps wrong.
     *  flipV = true for decoder-fed video quads: V is emitted in
     *  displayed-image space (v up) with the decoder flip left to uTexMat;
     *  canvas panels (plain sampler2D, no transform) use flipV = false. */
    private fun gridQuadP(
        p00: FloatArray, p10: FloatArray, p01: FloatArray, p11: FloatArray,
        nx: Int, ny: Int, flipV: Boolean = false
    ): Mesh {
        val verts = FloatArray((nx + 1) * (ny + 1) * 3)
        val texs = FloatArray((nx + 1) * (ny + 1) * 2)
        var vi = 0; var ti = 0
        for (iy in 0..ny) {
            val v = iy.toFloat() / ny
            for (ix in 0..nx) {
                val u = ix.toFloat() / nx
                for (k in 0..2) {
                    val top = p00[k] + (p10[k] - p00[k]) * u
                    val bot = p01[k] + (p11[k] - p01[k]) * u
                    verts[vi++] = top + (bot - top) * v
                }
                texs[ti++] = u; texs[ti++] = if (flipV) 1f - v else v
            }
        }
        val idx = mutableListOf<Short>()
        for (iy in 0 until ny) for (ix in 0 until nx) {
            val a = (iy * (nx + 1) + ix).toShort()
            val b = (a + 1).toShort(); val c = ((iy + 1) * (nx + 1) + ix).toShort(); val d = (c + 1).toShort()
            idx += listOf(a, c, b, b, c, d)
        }
        return Mesh(fb(verts), fb(texs), sb(idx.toShortArray()), idx.size)
    }

    private fun drawMesh2d(m: Mesh, texId: Int, mat: FloatArray, alpha: Float = 1f) {
        GLES20.glUseProgram(prog2d)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId)
        GLES20.glUniform1i(uTex2d, 0)
        GLES20.glUniform1f(uAlpha2d, alpha.coerceIn(0f, 1f))
        GLES20.glUniformMatrix4fv(uMvp2d, 1, false, mat, 0)
        GLES20.glEnableVertexAttribArray(aPos2d)
        GLES20.glVertexAttribPointer(aPos2d, 3, GLES20.GL_FLOAT, false, 0, m.verts)
        GLES20.glEnableVertexAttribArray(aTex2d)
        GLES20.glVertexAttribPointer(aTex2d, 2, GLES20.GL_FLOAT, false, 0, m.tex)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, m.indexCount, GLES20.GL_UNSIGNED_SHORT, m.indices)
        GLES20.glDisableVertexAttribArray(aPos2d)
        GLES20.glDisableVertexAttribArray(aTex2d)
    }

    private var browserGrid: Mesh? = null
    private var browserGridD = -1f
    private var browserGridEl = -999f
    private fun drawBrowser(alpha: Float = 1f) {
        maybeUploadBrowser()
        val d = panelDistM; val hw = panelHalfW(); val hh = panelHalfH()
        // rotation-only UI matrix: identical in both eyes, always fuses.
        // Grid cached: rebuilding it per frame churned direct buffers and
        // strobed the whole scene through GC. Panel floats at browserElevDeg
        // (0 = centered; elevated = below the play menu, facing viewer).
        val el = Math.toRadians(browserElevDeg.toDouble()).toFloat()
        if (browserGrid == null || browserGridD != d || browserGridEl != el) {
            val cx = 0f; val cy = (kotlin.math.sin(el) * d); val cz = (-kotlin.math.cos(el) * d)
            var nx = -cx; var ny = -cy; var nz = -cz
            val nl = kotlin.math.sqrt(nx * nx + ny * ny + nz * nz).coerceAtLeast(1e-6f)
            nx /= nl; ny /= nl; nz /= nl
            // up = n × (1,0,0) = (0, nz, -ny)
            val ux = 0f; val uy = nz; val uz = -ny
            fun corner(sx: Float, sy: Float) = floatArrayOf(
                cx + sx * hw + ux * sy * hh,
                cy + uy * sy * hh,
                cz + uz * sy * hh
            )
            browserGrid = gridQuadP(
                corner(-1f, 1f), corner(1f, 1f), corner(-1f, -1f), corner(1f, -1f), 12, 8
            )
            browserGridD = d; browserGridEl = el
            FileLog.i("LimpetVR-browser", "grid rebuild d=$d el=$el")
        }
        drawMesh2d(browserGrid!!, browserTexId, ovM, alpha)
    }

    private fun makeReticle(color: Int): Int {
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        val bmp = Bitmap.createBitmap(96, 96, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        // transparent background (needs BLEND enabled); small ring that
        // the draw code shrinks toward a point as dwell progresses
        c.drawColor(Color.TRANSPARENT)
        val p = Paint(Paint.ANTI_ALIAS_FLAG)
        p.color = color; p.style = Paint.Style.STROKE; p.strokeWidth = 7f
        c.drawCircle(48f, 48f, 30f, p)
        p.style = Paint.Style.FILL
        c.drawCircle(48f, 48f, 5f, p)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        bmp.recycle()
        return tex[0]
    }

    /** Shaping preview cell: distorted 9x9 grid + moved dots (amber→red by
     *  magnitude) + displacement vectors + minimum convex polygon of moved
     *  points. Box is 56px at row left; grid coords [0,1], y down. */
    private fun drawShapePreview(c: Canvas, p: Paint, r: BrowserRow, y: Int) {
        val mags = r.previewMags ?: return
        val pos = r.previewPos ?: return
        val n = r.previewN.coerceAtLeast(2)
        if (mags.size < n * n || pos.size < n * n * 2) return
        val bx0 = 28f; val by0 = y.toFloat() + 4f; val bs = 56f
        fun px(j: Int) = bx0 + pos[j * 2].toFloat().coerceIn(-0.2f, 1.2f) * bs
        fun py(j: Int) = by0 + pos[j * 2 + 1].toFloat().coerceIn(-0.2f, 1.2f) * bs
        fun nx(j: Int) = bx0 + (j % n).toFloat() / (n - 1) * bs
        fun ny(j: Int) = by0 + (j / n).toFloat() / (n - 1) * bs
        // convex hull fill + stroke
        val hull = r.previewHull
        if (hull != null && hull.size >= 3) {
            val path = Path()
            path.moveTo(px(hull[0]), py(hull[0]))
            for (k in 1 until hull.size) path.lineTo(px(hull[k]), py(hull[k]))
            path.close()
            p.style = Paint.Style.FILL; p.color = Color.argb(40, 8, 145, 178)
            c.drawPath(path, p)
            p.style = Paint.Style.STROKE; p.strokeWidth = 2f; p.color = Color.rgb(8, 145, 178)
            c.drawPath(path, p)
            p.style = Paint.Style.FILL; p.strokeWidth = 1f
        }
        // distorted grid lines
        p.style = Paint.Style.STROKE; p.strokeWidth = 1f; p.color = Color.rgb(150, 150, 150)
        for (i in 0 until n) {
            val rowPath = Path()
            rowPath.moveTo(px(i * n), py(i * n))
            for (j in 1 until n) rowPath.lineTo(px(i * n + j), py(i * n + j))
            c.drawPath(rowPath, p)
            val colPath = Path()
            colPath.moveTo(px(i), py(i))
            for (k in 1 until n) colPath.lineTo(px(k * n + i), py(k * n + i))
            c.drawPath(colPath, p)
        }
        p.style = Paint.Style.FILL
        // displacement vectors (nominal -> offset), faint (style still STROKE)
        p.color = Color.argb(120, 125, 211, 252); p.strokeWidth = 1f
        for (j in mags.indices) {
            if (mags[j] <= 1e-6f) continue
            c.drawLine(nx(j), ny(j), px(j), py(j), p)
        }
        p.style = Paint.Style.FILL
        // dots like shapemesh.py: grey unmoved; moved toward the centre
        // (dot of displacement vs to-centre vector > eps) green, else red.
        // Sign-based, so grid-space (y down) works unchanged.
        for (j in mags.indices) {
            val m = mags[j].coerceIn(0f, 1f)
            val x = px(j); val yy = py(j)
            if (m <= 1e-6f) {
                p.color = Color.rgb(170, 170, 170)
                c.drawCircle(x, yy, 2f, p)
            } else {
                val ix = (j % n).toFloat() / (n - 1)
                val iy = (j / n).toFloat() / (n - 1)
                val dx = pos[j * 2].toFloat() - ix
                val dy = pos[j * 2 + 1].toFloat() - iy
                val cx = 0.5f - ix; val cy = 0.5f - iy
                val t = m.coerceAtLeast(0.15f)
                val inward = (dx * dx + dy * dy) > 1e-18f &&
                    (cx * cx + cy * cy) > 1e-18f && (dx * cx + dy * cy) > 1e-9f
                if (inward)
                    p.color = Color.rgb((120 - 90 * t).toInt(), (200 - 20 * t).toInt(), (120 - 90 * t).toInt())
                else
                    p.color = Color.rgb(255, (200 - 170 * t).toInt(), (60 - 40 * t).toInt())
                c.drawCircle(x, yy, 2f + 4f * t, p)
            }
        }
        p.strokeWidth = 1f
    }

    private fun maybeUploadBrowser() {
        ensureVisible()
        val rows = browserRows
        val pin = pinTopRows.coerceIn(0, 2)
        val stm = pin > 0 || rows.size > VISIBLE_ROWS // strips mode
        val win = if (stm) winRows(pin) else VISIBLE_ROWS
        val base = scrollPos.toInt()
        val winStart = (pin + base).coerceAtMost(rows.size)
        val winEnd = (winStart + win + (if (stm) 1 else 0)).coerceAtMost(rows.size)
        var h = browserTitle.hashCode() * 31 + (if (stm) (scrollPos * ROW_H).toInt() else base) + pin * 7919
        for (i in 0 until pin.coerceAtMost(rows.size)) h = h * 31 + rowHash(rows, i)
        for (i in winStart until winEnd) h = h * 31 + rowHash(rows, i)
        h = h * 31 + highlight + (if (webSideBtns) 20011 else 0) +
            (sideBtnDir * 7919 + (sideBtnProg * 255).toInt()) +
            (if (sliderHoverU >= 0f) (sliderHoverU * 128).toInt() else 0)
        if (stm) h += scrollEngage * 131071 + scrollTrigDir * 1031 + (scrollTrigF * 32).toInt()
        if (h == lastPanelHash && browserBitmap != null) return
        lastPanelHash = h
        val bmp = Bitmap.createBitmap(TEX, TEX, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.rgb(13, 20, 28))
        val p = Paint(Paint.ANTI_ALIAS_FLAG)
        p.color = Color.WHITE; p.textSize = 44f
        // Don't draw the web title under the PgUp/PgDn buttons.
        val titleMax = if (webSideBtns) 21 else 30
        c.drawText(browserTitle.take(titleMax), 40f, 72f, p)
        if (webSideBtns) {
            p.textSize = 26f; p.textAlign = Paint.Align.CENTER
            val upProg = if (sideBtnDir == -1 && !sideBtnFired) sideBtnProg else 0f
            val dnProg = if (sideBtnDir == 1 && !sideBtnFired) sideBtnProg else 0f
            drawSideBtn(c, p, SIDE_BTN_UP_Y0, "PgUp", sideBtnDir == -1, upProg)
            drawSideBtn(c, p, SIDE_BTN_DN_Y0, "PgDn", sideBtnDir == 1, dnProg)
            p.textAlign = Paint.Align.LEFT; p.textSize = 44f
        }
        if (stm) {
            // File pages: pinned nav rows, scroll strips, fractional window.
            val upY0 = upStripY0(pin); val rY0 = rowsY0(pin); val dnY0 = downStripY0(pin)
            val scrollable = rows.size > pin + win
            var py = PIN_Y0.toFloat()
            for (i in 0 until pin.coerceAtMost(rows.size)) {
                drawBrowserRow(c, p, rows, i, py.toInt())
                py += ROW_H
            }
            if (scrollable) drawScrollStrip(c, p, upY0, -1)
            val fracPx = scrollPos * ROW_H - base * ROW_H
            c.save()
            c.clipRect(0f, rY0, TEX.toFloat(), rY0 + win * ROW_H)
            c.translate(0f, -fracPx)
            var y = rY0
            for (i in winStart until winEnd) {
                drawBrowserRow(c, p, rows, i, y.toInt())
                y += ROW_H
            }
            c.restore()
            if (scrollable) drawScrollStrip(c, p, dnY0, 1)
        } else {
            // Settings pages: plain fixed window, no strips.
            var y = ROWS_Y0
            for (i in winStart until winEnd) {
                drawBrowserRow(c, p, rows, i, y)
                y += ROW_H
            }
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, browserTexId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        browserBitmap?.recycle()
        browserBitmap = bmp
    }

    /** Hash contribution of one browser row (matches what drawBrowserRow paints). */
    private fun rowHash(rows: List<BrowserRow>, i: Int): Int {
        var h = rows[i].label.hashCode() * 7 + rows[i].meta.hashCode() + rows[i].segSelected
        // shaping previews change with weights/toggles: sample magnitudes into the hash
        val pm = rows[i].previewMags
        if (pm != null) {
            var j = 0
            while (j < pm.size) { h = h * 31 + (pm[j] * 1000).toInt(); j += 7 }
            h = h * 31 + (rows[i].previewHull?.size ?: 0)
        }
        return h
    }

    /** Pinned scroll strip (file pages only): wide bar with
     *  trigger-progress fill while earning engagement, solid while
     *  gliding, dimmed at the travel end. */
    /** One PgUp/PgDn button; fills as the dwell completes. */
    private fun drawSideBtn(c: Canvas, p: Paint, y0: Float, label: String,
                            hot: Boolean, prog: Float) {
        p.color = when {
            hot -> Color.rgb(30, 58, 95)
            else -> Color.rgb(21, 32, 45)
        }
        c.drawRect(SIDE_BTN_X0, y0, SIDE_BTN_X1, y0 + SIDE_BTN_H, p)
        if (prog > 0f) {
            p.color = Color.rgb(8, 145, 178)
            c.drawRect(SIDE_BTN_X0, y0, SIDE_BTN_X0 + (SIDE_BTN_X1 - SIDE_BTN_X0) * prog,
                y0 + SIDE_BTN_H, p)
        }
        p.color = if (hot) Color.WHITE else Color.rgb(148, 163, 184)
        p.textSize = 26f
        c.drawText(label, (SIDE_BTN_X0 + SIDE_BTN_X1) * 0.5f, y0 + 32f, p)
    }

    private fun drawScrollStrip(c: Canvas, p: Paint, y0: Float, dir: Int) {
        val pin = pinTopRows.coerceIn(0, 2)
        val maxS = maxOf(0, browserRows.size - pin - winRows(pin)).toFloat()
        val atEnd = (dir < 0 && scrollPos <= 0f) || (dir > 0 && scrollPos >= maxS)
        val active = scrollEngage == dir
        p.color = when {
            atEnd -> Color.rgb(30, 41, 55)
            active -> Color.rgb(8, 145, 178)
            else -> Color.rgb(30, 58, 95)
        }
        c.drawRect(20f, y0, 1004f, y0 + STRIP_H, p)
        val prog = if (scrollTrigDir == dir && !active) scrollTrigF.coerceIn(0f, 1f) else 0f
        if (prog > 0f) {
            p.color = Color.rgb(8, 145, 178)
            c.drawRect(20f, y0, 20f + 984f * prog, y0 + STRIP_H, p)
        }
        p.color = if (atEnd) Color.rgb(100, 116, 139) else Color.WHITE
        p.textSize = 30f; p.textAlign = Paint.Align.CENTER
        val g = if (dir < 0) "▲" else "▼"
        c.drawText("$g  scroll ${if (dir < 0) "up" else "down"}  $g", 512f, y0 + 38f, p)
        p.textAlign = Paint.Align.LEFT
    }

    /** Single browser list row at panel y (int, TEX coords). */
    private fun drawBrowserRow(c: Canvas, p: Paint, rows: List<BrowserRow>, i: Int, y: Int) {
            val r = rows[i]
            if (i == highlight) {
                p.color = Color.rgb(30, 58, 95)
                c.drawRect(20f, y.toFloat(), 1004f, (y + ROW_H).toFloat(), p)
            }
            if (r.previewMags != null && r.previewPos != null) {
                // shaping preview row: mini 9x9 grid with moved dots, hull + vectors
                drawShapePreview(c, p, r, y)
                p.color = Color.WHITE; p.textSize = 30f; p.textAlign = Paint.Align.LEFT
                c.drawText(r.label.take(24), 100f, (y + 36).toFloat(), p)
                if (r.meta.isNotEmpty()) {
                    p.color = Color.rgb(148, 163, 184); p.textSize = 20f
                    c.drawText(r.meta.take(40), 100f, (y + 58).toFloat(), p)
                }
            } else if (r.dead) {
                // rest zone: thin divider, nothing to activate
                p.color = Color.rgb(51, 65, 85)
                c.drawRect(44f, (y + ROW_H / 2 - 1).toFloat(), 1000f, (y + ROW_H / 2 + 1).toFloat(), p)
            } else if (r.segLabels.isNotEmpty()) {
                // segmented button row: N equal buttons across the row width
                val n = r.segLabels.size
                val x0 = 20f; val x1 = 1004f
                val bw = (x1 - x0) / n
                p.textSize = 30f; p.textAlign = Paint.Align.CENTER
                for (s in 0 until n) {
                    val sx0 = x0 + s * bw + 3f
                    val sx1 = x0 + (s + 1) * bw - 3f
                    if (s == r.segSelected) {
                        p.color = Color.rgb(8, 145, 178)
                        c.drawRect(sx0, y.toFloat() + 6f, sx1, (y + ROW_H - 6).toFloat(), p)
                        p.color = Color.WHITE
                    } else {
                        p.color = Color.rgb(51, 65, 85)
                        c.drawRect(sx0, y.toFloat() + 6f, sx1, (y + ROW_H - 6).toFloat(), p)
                        p.color = Color.rgb(203, 213, 225)
                    }
                    c.drawText(r.segLabels[s].take(12), (sx0 + sx1) / 2f, (y + 41).toFloat(), p)
                }
                // live tooltip: name of the segment the gaze would select
                if (i == highlight && sliderHoverU >= 0f) {
                    val fx = ((sliderHoverU * 1024f - 20f) / 984f).coerceIn(0f, 0.999f)
                    val seg = (fx * n).toInt().coerceIn(0, n - 1)
                    val txt = r.segLabels[seg].take(12)
                    val cx = x0 + (seg + 0.5f) * bw
                    p.textSize = 20f
                    val tw = p.measureText(txt)
                    val bx0 = (cx - tw / 2f - 10f).coerceAtLeast(24f)
                    val bx1 = (cx + tw / 2f + 10f).coerceAtMost(1000f)
                    p.color = Color.rgb(10, 14, 22)
                    c.drawRect(bx0, (y + 18).toFloat(), bx1, (y + 46).toFloat(), p)
                    p.color = Color.WHITE
                    p.style = Paint.Style.STROKE; p.strokeWidth = 3f
                    c.drawRect(bx0, (y + 18).toFloat(), bx1, (y + 46).toFloat(), p)
                    p.style = Paint.Style.FILL
                    c.drawText(txt, (bx0 + bx1) / 2f, (y + 39).toFloat(), p)
                }
                p.textAlign = Paint.Align.LEFT
            } else {
            if (r.slideKey == null) {
                val icon = when (r.kind) {
                    BrowserRow.FOLDER -> "📁"
                    BrowserRow.VIDEO -> "🎬"
                    BrowserRow.ACTION -> "⚙"
                    else -> "📄"
                }
                p.color = Color.WHITE; p.textSize = 36f
                c.drawText("$icon  ${r.label.take(30)}", 44f, (y + 34).toFloat(), p)
            }
            if (r.slideKey != null) {
                // gaze slider (compact): label + value on top line, bar
                // mid-row, live tooltip bubble below the bar at the gaze
                // position showing the value a dwell would select
                val frac = ((r.slideVal - r.slideMin) / (r.slideMax - r.slideMin)).coerceIn(0f, 1f)
                p.color = Color.WHITE; p.textSize = 28f
                c.drawText(r.label.take(30), 44f, (y + 26).toFloat(), p)
                p.color = Color.rgb(125, 211, 252); p.textSize = 20f; p.textAlign = Paint.Align.RIGHT
                c.drawText(r.meta.take(20), 1000f, (y + 26).toFloat(), p)
                p.textAlign = Paint.Align.LEFT
                p.color = Color.rgb(51, 65, 85)
                c.drawRect(44f, (y + 30).toFloat(), 1000f, (y + 42).toFloat(), p)
                p.color = Color.rgb(125, 211, 252)
                c.drawRect(44f, (y + 30).toFloat(), 44f + 956f * frac, (y + 42).toFloat(), p)
                if (i == highlight && sliderHoverU >= 0f) {
                    val hf = barFrac(sliderHoverU)
                    var rraw = r.slideMin + hf * (r.slideMax - r.slideMin)
                    val fm = r.slideFmt
                    if (fm != null && fm.snap > 0f) rraw = Math.round(rraw / fm.snap).toFloat() * fm.snap
                    val disp = rraw * (fm?.scale ?: 1f) + (fm?.offset ?: 0f)
                    val dec = fm?.decimals ?: 0
                    val num = if (dec == 0) Math.round(disp).toString()
                        else String.format(Locale.US, "%.${dec}f", disp)
                    val txt = num + (fm?.suffix ?: "")
                    val tx = (44f + hf * 956f).coerceIn(70f, 954f)
                    p.textSize = 16f; p.textAlign = Paint.Align.CENTER
                    val tw = p.measureText(txt)
                    val bx0 = (tx - tw / 2f - 10f).coerceAtLeast(24f)
                    val bx1 = (tx + tw / 2f + 10f).coerceAtMost(1000f)
                    p.color = Color.rgb(10, 14, 22)
                    c.drawRect(bx0, (y + 44).toFloat(), bx1, (y + 62).toFloat(), p)
                    p.color = Color.rgb(125, 211, 252)
                    p.style = Paint.Style.STROKE; p.strokeWidth = 2f
                    c.drawRect(bx0, (y + 44).toFloat(), bx1, (y + 62).toFloat(), p)
                    p.style = Paint.Style.FILL
                    p.color = Color.WHITE
                    c.drawText(txt, (bx0 + bx1) / 2f, (y + 58).toFloat(), p)
                    p.textAlign = Paint.Align.LEFT
                }
            } else if (r.meta.isNotEmpty()) {
                p.color = Color.rgb(148, 163, 184); p.textSize = 24f
                c.drawText("    ${r.meta.take(56)}", 44f, (y + 58).toFloat(), p)
            }
            }
    }

    override fun onFrameAvailable(st: SurfaceTexture?) { frameAvailable = true; arrivedFrames++ }

    // ---------- play menu drawing ----------
    private var menuGrid: Mesh? = null
    private var menuGridD = -1f
    private var menuGridEl = -999f
    private fun drawMenuPanel(alpha: Float = 1f) {
        maybeUploadMenu()
        // World-locked plane at radius panelDistM whose elevation animates
        // through the ⇅ flip (the grid rebuilds each frame mid-flip, then
        // settles and is cached). The rect is authored in DESIGN space —
        // MENU_X0..X1 × MENU_Y0..Y1, tuned for MENU_DESIGN_R — and every
        // extent is multiplied by menuScale(), so the panel keeps its
        // apparent size while the distance slider only changes its depth.
        val d = panelDistM
        val el = Math.toRadians(menuElevCurrent().toDouble()).toFloat()
        val k = menuScale()
        if (menuGrid == null || menuGridD != d || menuGridEl != el) {
            val cx = 0f; val cy = (kotlin.math.sin(el) * d); val cz = (-kotlin.math.cos(el) * d)
            var nx = -cx; var ny = -cy; var nz = -cz
            val nl = kotlin.math.sqrt(nx * nx + ny * ny + nz * nz).coerceAtLeast(1e-6f)
            nx /= nl; ny /= nl; nz /= nl
            // up = n × (1,0,0) = (0, nz, -ny)
            val ux = 0f; val uy = nz; val uz = -ny
            fun corner(dx: Float, dy: Float) = floatArrayOf(
                cx + dx * k + ux * dy * k,
                cy + uy * dy * k,
                cz + uz * dy * k
            )
            menuGrid = gridQuadP(
                corner(MENU_X0, MENU_Y1), corner(MENU_X1, MENU_Y1),
                corner(MENU_X0, MENU_Y0), corner(MENU_X1, MENU_Y0),
                16, 6
            )
            menuGridD = d; menuGridEl = el
        }
        drawMesh2d(menuGrid!!, menuTexId, ovM, alpha)
    }

    /** One bitmap for the whole panel: backdrop, title, transport row,
     *  seek bar, the right-hand zoom/fov/volume columns and the transport
     *  row — all in design units mapped to texels, re-uploaded on the GL thread
     *  whenever state changes. Alpha follows doc §7.4: gazed brightens,
     *  dwell focus dims (1 − 0.5·progress), backdrop sits at 0.3. */
    private fun maybeUploadMenu() {
        val posSec = (menuPosMs / 1000).toInt()
        val durSec = (menuDurMs / 1000).toInt()
        val flashing = menuFlash.isNotEmpty() && now() < menuFlashUntil
        // Every button dims with its own dwell, so progress is part of the
        // cache key (quantised: it moves every frame while a dwell runs).
        var dwellSum = 0
        for (i in menuProg.indices) dwellSum += (menuProg[i] * 64f).toInt()
        val h = menuHighlight * 31 + posSec * 131 + durSec * 17 +
            (if (menuPlaying) 1 else 0) + (if (flashing) 1009 else 0) + menuFlash.hashCode() +
            (if (menuSeekHoverU >= 0f) (menuSeekHoverU * 128).toInt() else 0) +
            menuTitle.hashCode() * 7 + skipSecs + dwellSum
        if (h == lastMenuHash && menuBitmap != null) return
        lastMenuHash = h
        val W = MENU_TEX_W; val H = MENU_TEX_H
        val bmp = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.TRANSPARENT)
        val p = Paint(Paint.ANTI_ALIAS_FLAG)
        p.style = Paint.Style.FILL
        // design -> texel: bitmap y grows downward, design y grows up
        val sx = W / (MENU_X1 - MENU_X0)
        val sy = H / (MENU_Y1 - MENU_Y0)
        fun tx(x: Float) = (x - MENU_X0) * sx
        fun ty(y: Float) = (MENU_Y1 - y) * sy
        fun box(b: MenuBtn) = floatArrayOf(
            tx(b.x - b.hw), ty(b.y + b.hh), tx(b.x + b.hw), ty(b.y - b.hh)
        )
        fun white(a: Int) = Color.argb(a, 255, 255, 255)
        /** §7.4 alpha: gazed 1.0, idle 0.9, times (1 − 0.5·dwell). */
        fun alphaOf(id: Int): Int {
            val base = if (id == menuHighlight) 1f else 0.9f
            val focus = if (id != -2) menuProg[menuSlot(id)].coerceIn(0f, 1f) else 0f
            return (255f * base * (1f - 0.5f * focus)).toInt().coerceIn(0, 255)
        }
        fun text(s: String, cx: Float, cy: Float, size: Float, a: Int, outline: Boolean = false) {
            p.textSize = size; p.textAlign = Paint.Align.CENTER
            val by = cy + size * 0.35f
            if (outline) {
                // black rim first so the white glyphs stay legible over
                // any frame behind the (transparent) panel edge
                p.style = Paint.Style.STROKE; p.strokeWidth = size / 5f
                p.color = Color.argb(a, 0, 0, 0)
                c.drawText(s, cx, by, p)
            }
            p.style = Paint.Style.FILL
            p.color = white(a)
            c.drawText(s, cx, by, p)
            p.textAlign = Paint.Align.LEFT
        }
        // backdrop pane first: everything else draws over it (§7.3)
        val bd = box(menuBackdrop)
        p.color = Color.argb(77, 184, 188, 196)
        c.drawRect(bd[0], bd[1], bd[2], bd[3], p)
        // file name across the top strip
        val tr = box(menuTitleRect)
        p.color = white(255)
        p.textSize = 44f; p.textAlign = Paint.Align.CENTER
        val title = menuTitle.take(48)
        while (p.textSize > 18f && p.measureText(title) > (tr[2] - tr[0] - 24f)) p.textSize -= 2f
        c.drawText(title, (tr[0] + tr[2]) / 2f, (tr[1] + tr[3]) / 2f + p.textSize * 0.35f, p)
        p.textAlign = Paint.Align.LEFT
        // ---- buttons ----
        fun highlightBox(b: MenuBtn, a: Int) {
            if (b.id != menuHighlight) return
            val r = box(b)
            p.color = Color.argb(a, 30, 58, 95)
            c.drawRect(r[0], r[1], r[2], r[3], p)
        }
        for (b in menuButtons) {
            val a = alphaOf(b.id)
            val cx = tx(b.x); val cy = ty(b.y)
            highlightBox(b, a)
            when (b.id) {
                // transport / utility row: emoji-ish text glyphs
                5 -> text(if (menuPlaying) "⏸" else "▶", cx, cy, 44f, a)
                8 -> magnifierGlyph(c, p, cx, cy, 40f, a, true)
                9 -> magnifierGlyph(c, p, cx, cy, 40f, a, false)
                10 -> speakerGlyph(c, p, cx, cy, 13f, a, true)
                11 -> speakerGlyph(c, p, cx, cy, 13f, a, false)
                14 -> fovGlyph(c, p, cx, cy, 40f, a, false)
                15 -> fovGlyph(c, p, cx, cy, 40f, a, true)
                13 -> crosshairGlyph(c, p, cx, cy, 40f, a)
                12 -> flipGlyph(c, p, cx, cy, 40f, a)
                else -> text(b.glyph, cx, cy, 44f, a)
            }
        }
        // ---- seek bar ----
        val bar = box(menuBar)
        val barA = alphaOf(-1)
        p.color = Color.argb(barA, 51, 65, 85)
        c.drawRect(bar[0], bar[1], bar[2], bar[3], p)
        val frac = if (menuDurMs > 0) (menuPosMs.toFloat() / menuDurMs).coerceIn(0f, 1f) else 0f
        p.color = Color.argb(barA, 125, 211, 250)
        c.drawRect(bar[0], bar[1], bar[0] + (bar[2] - bar[0]) * frac, bar[3], p)
        if (menuHighlight == -1) {
            p.color = white(barA); p.style = Paint.Style.STROKE; p.strokeWidth = 4f
            c.drawRect(bar[0], bar[1], bar[2], bar[3], p)
            p.style = Paint.Style.FILL
        }
        // position / duration under the bar (the live seek time rides the
        // head-locked tooltip pill instead, so the two never overlap)
        if (!(menuHighlight == -1 && menuSeekHoverU >= 0f && menuDurMs > 0)) {
            text(
                if (flashing) menuFlash else "${fmtTime(menuPosMs)} / ${fmtTime(menuDurMs)}",
                tx(menuBar.x), ty(-1.5f), 24f, 255, outline = true
            )
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, menuTexId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        menuBitmap?.recycle()
        menuBitmap = bmp
    }

    // ---- menu glyphs (white on transparent; `a` = 0..255 alpha) ----
    /** Magnifier with + / − inside the lens (emoji turns to mush small). */
    private fun magnifierGlyph(c: Canvas, p: Paint, cx: Float, cy: Float, s: Float, a: Int, plus: Boolean) {
        val r = s * 0.30f
        val lx = cx - r * 0.35f; val ly = cy - r * 0.25f
        val sw = (s * 0.09f).coerceAtLeast(3f)
        p.color = Color.argb(a, 255, 255, 255); p.style = Paint.Style.STROKE; p.strokeWidth = sw
        c.drawCircle(lx, ly, r, p)
        val hx = lx + r * 0.72f; val hy = ly + r * 0.72f
        c.drawLine(hx, hy, hx + r * 0.85f, hy + r * 0.85f, p)
        p.style = Paint.Style.FILL
        val bw = r * 1.1f
        c.drawRect(lx - bw / 2f, ly - sw / 2f, lx + bw / 2f, ly + sw / 2f, p)
        if (plus) c.drawRect(lx - sw / 2f, ly - bw / 2f, lx + sw / 2f, ly + bw / 2f, p)
        p.style = Paint.Style.FILL
    }
    /** Speaker body; loud adds the second wave (identical geometry both). */
    private fun speakerGlyph(c: Canvas, p: Paint, cx: Float, cy: Float, s: Float, a: Int, loud: Boolean) {
        val col = Color.argb(a, 255, 255, 255)
        val sw = (s * 0.09f).coerceAtLeast(3f)
        val bx1 = cx - s * 0.35f
        p.color = col; p.style = Paint.Style.FILL
        c.drawRect(cx - s * 1.1f, cy - s * 0.55f, bx1, cy + s * 0.55f, p)
        val tipX = cx + s * 0.25f
        c.drawPath(Path().apply {
            moveTo(bx1, cy - s * 0.55f); lineTo(tipX, cy - s)
            lineTo(tipX, cy + s); lineTo(bx1, cy + s * 0.55f); close()
        }, p)
        p.style = Paint.Style.STROKE; p.strokeWidth = sw
        fun wave(rr: Float) = c.drawArc(tipX - rr, cy - rr, tipX + rr, cy + rr, -55f, 110f, false, p)
        wave(s * 0.62f)
        if (loud) wave(s * 1.12f)
        p.style = Paint.Style.FILL
    }
    /** Screen frame with − / +: FOV narrower / wider. */
    private fun fovGlyph(c: Canvas, p: Paint, cx: Float, cy: Float, s: Float, a: Int, plus: Boolean) {
        val col = Color.argb(a, 255, 255, 255)
        val sw = (s * 0.09f).coerceAtLeast(3f)
        val fw = s * 0.46f; val fh = s * 0.32f
        p.color = col; p.style = Paint.Style.STROKE; p.strokeWidth = sw
        c.drawRect(cx - fw, cy - fh, cx + fw, cy + fh, p)
        val b = s * 0.30f
        c.drawLine(cx - b, cy, cx + b, cy, p)
        if (plus) c.drawLine(cx, cy - b, cx, cy + b, p)
        p.style = Paint.Style.FILL
    }
    /** Recenter crosshair: stroked ring + center dot. */
    private fun crosshairGlyph(c: Canvas, p: Paint, cx: Float, cy: Float, s: Float, a: Int) {
        val r = s * 0.30f
        val sw = (s * 0.09f).coerceAtLeast(3f)
        p.color = Color.argb(a, 255, 255, 255); p.style = Paint.Style.STROKE; p.strokeWidth = sw
        c.drawCircle(cx, cy, r, p)
        p.style = Paint.Style.FILL
        c.drawCircle(cx, cy, sw, p)
    }
    /** Menu-side flip: up chevron over down chevron (the old ⇅ text glyph,
     *  drawn vector so it can't tofu). */
    private fun flipGlyph(c: Canvas, p: Paint, cx: Float, cy: Float, s: Float, a: Int) {
        val col = Color.argb(a, 255, 255, 255)
        val w = s * 0.22f; val h = s * 0.16f; val gap = s * 0.07f
        p.color = col; p.style = Paint.Style.FILL
        c.drawPath(Path().apply {
            moveTo(cx - w, cy - gap); lineTo(cx + w, cy - gap); lineTo(cx, cy - gap - h); close()
        }, p)
        c.drawPath(Path().apply {
            moveTo(cx - w, cy + gap); lineTo(cx + w, cy + gap); lineTo(cx, cy + gap + h); close()
        }, p)
    }


    private fun fmtTime(ms: Long): String {
        val s = (ms / 1000).toInt().coerceAtLeast(0)
        return "%d:%02d".format(s / 60, s % 60)
    }

    // ---- meshes ----
    private data class Mesh(val verts: FloatBuffer, val tex: FloatBuffer, val indices: java.nio.ShortBuffer, val indexCount: Int)

    /** One enabled shaping grid: offsets in half-frame normalized units
     *  (ox +right, oy +down/grid-space), weight 0..1. Arrays never mutated
     *  after creation; the whole list is replaced atomically (volatile). */
    data class ActiveShape(val ox: FloatArray, val oy: FloatArray, val weight01: Float, val n: Int)
    @Volatile var shapingActive: List<ActiveShape> = emptyList()
    @Volatile var shapingRevision: Int = 0

    /** Curved flat screen (screenCurve 0..1): every vertex blends the flat
     *  quad's point with the eye-centred spherical-cap point of radius Rt,
     *  so curve 0 reproduces the plane exactly and curve 1 bends the whole
     *  picture around the viewer (IMAX-style wrap, both axes — the top edge
     *  swings forward too, which is what kills the stretch of a huge flat
     *  screen). World size (w × h) is baked in because an eye-centred cap
     *  cannot be scaled to change apparent size, and the wrap angle w/Rt is
     *  clamped to SCREEN_ARC_MAX_DEG so the edges stay in front of the ears
     *  (this is a curved screen, not a dome). Texcoords and winding match
     *  gridQuadP(flipV = true) so the FLAT draw path is untouched. */
    private fun screenCapMesh(curve: Float, w: Float, h: Float): Mesh {
        val c = curve.coerceIn(0f, 1f)
        val arcMax = Math.toRadians(SCREEN_ARC_MAX_DEG.toDouble()).toFloat()
        val rt = maxOf(SCREEN_R_MIN, w / arcMax)
        val cols = 48; val rows = 24
        val verts = FloatArray((cols + 1) * (rows + 1) * 3)
        val texs = FloatArray((cols + 1) * (rows + 1) * 2)
        var vi = 0; var ti = 0
        for (iy in 0..rows) {
            val v = iy.toFloat() / rows
            val by = (0.5f - v) * h / rt
            val cy = kotlin.math.cos(by); val sy = kotlin.math.sin(by)
            val fy = (0.5f - v) * h
            for (ix in 0..cols) {
                val u = ix.toFloat() / cols
                val ax = (u - 0.5f) * w / rt
                val fx = (u - 0.5f) * w
                val cx = rt * cy * kotlin.math.sin(ax)
                val ry = rt * sy
                val cz = -rt * cy * kotlin.math.cos(ax)
                verts[vi++] = fx + (cx - fx) * c
                verts[vi++] = fy + (ry - fy) * c
                verts[vi++] = -FLAT_DIST + (cz + FLAT_DIST) * c
                texs[ti++] = u; texs[ti++] = 1f - v
            }
        }
        val idx = mutableListOf<Short>()
        for (iy in 0 until rows) for (ix in 0 until cols) {
            val a = (iy * (cols + 1) + ix).toShort()
            val b = (a + 1).toShort(); val cc = ((iy + 1) * (cols + 1) + ix).toShort(); val d = (cc + 1).toShort()
            idx += listOf(a, cc, b, b, cc, d)
        }
        FileLog.i("LimpetVR-GL", "flat cap built curve=${"%.2f".format(c)} rt=${"%.2f".format(rt)} " +
            "arc=${"%.0f".format(Math.toDegrees((w / rt).toDouble()))}x" +
            "${"%.0f".format(Math.toDegrees((h / rt).toDouble()))}deg " +
            "w=${"%.2f".format(w)} h=${"%.2f".format(h)} verts=${verts.size / 3}")
        return Mesh(fb(verts), fb(texs), sb(idx.toShortArray()), idx.size)
    }

    private fun buildMesh(proj: Projection): Mesh {
        val m = when (proj) {
            // FLAT: curve 0 = unit square at the origin — buildVideoModel()
            // puts it in place — T(0, 0.25, −11.95) · S(w/2, h/2, 1) — so the
            // mesh itself carries no size or aspect (§5: the plane's world
            // width is 8.5·screenSize, aspect-corrected per eye). curve > 0
            // bends it: see screenCapMesh(), whose vertices already carry
            // world size, so the model matrix only lifts to eye height.
            Projection.FLAT -> if (screenCurve > 0f) {
                val (w, h) = screenDims()
                screenCapMesh(screenCurve, w, h)
            } else gridQuadP(
                floatArrayOf(-1f, 1f, 0f), floatArrayOf(1f, 1f, 0f),
                floatArrayOf(-1f, -1f, 0f), floatArrayOf(1f, -1f, 0f),
                24, 12, flipV = true
            )
            // Fisheye uses the same 180° sphere mesh as the domes (§8: mesh,
            // basis, projection and zoom identical — only sampling differs).
            // Mirrored rigs are handled in the circle lookup, not the mesh.
            Projection.FISHEYE -> sphereSegment(180f)
            Projection.DEG180 -> sphereSegment(180f)
            Projection.DEG220 -> sphereSegment(220f)
            Projection.DEG270 -> sphereSegment(270f)
            Projection.DEG360 -> sphereSegment(360f)
        }
        return bakeShaping(m)
    }

    /** Bake the normalized weighted-average shaping grid into the video mesh
     *  UVs (texture space, so head tracking via MVP is unaffected). Mesh UVs
     *  are per-half-frame: u right, v up (GL origin). Grid offsets are
     *  DISPLAY displacements (where content goes, y down) but sampling
     *  needs the opposite: to move content toward center you sample from
     *  outside, so the write NEGATES (Skinny authored narrower rendered
     *  wider before this fix). Grid space is y down, so grid_v = 1 - v
     *  and the y write is v + dy. No-op when nothing enabled. */
    private fun bakeShaping(m: Mesh): Mesh {
        val active = shapingActive
        if (active.isEmpty()) return m
        val n = active[0].n
        var wsum = 0f; var wcnt = 0
        for (a in active) if (a.n == n) { wsum += a.weight01; wcnt++ }
        if (wsum <= 0f) return m
        // Combine on the fly per vertex (meshes are small: 61x49 sphere, 25x13 flat).
        val count = m.tex.capacity() / 2
        val out = FloatArray(m.tex.capacity())
        m.tex.rewind()
        for (k in 0 until count) {
            val u = m.tex.get()
            val v = m.tex.get()
            // bilinear sample of averaged offsets at grid coords (u, 1-v)
            var dx = 0f; var dy = 0f
            val gx = (u.coerceIn(0f, 1f) * (n - 1)).coerceIn(0f, (n - 1).toFloat())
            val gv = ((1f - v).coerceIn(0f, 1f) * (n - 1)).coerceIn(0f, (n - 1).toFloat())
            val x0 = gx.toInt().coerceAtMost(n - 2); val y0 = gv.toInt().coerceAtMost(n - 2)
            val fx = gx - x0; val fy = gv - y0
            for (a in active) {
                if (a.n != n) continue
                // MEAN of weighted offsets (not normalized average): weights
                // are absolute opacities, so a lone shape at 50% gives half
                // displacement. (Normalized average pinned every nonzero
                // weight to full strength — the slider did nothing.)
                val w = a.weight01 / wcnt
                fun at(ix: Int, iy: Int, arr: FloatArray) = arr[iy * n + ix]
                val ox = (at(x0, y0, a.ox) * (1 - fx) + at(x0 + 1, y0, a.ox) * fx) * (1 - fy) +
                         (at(x0, y0 + 1, a.ox) * (1 - fx) + at(x0 + 1, y0 + 1, a.ox) * fx) * fy
                val oy = (at(x0, y0, a.oy) * (1 - fx) + at(x0 + 1, y0, a.oy) * fx) * (1 - fy) +
                         (at(x0, y0 + 1, a.oy) * (1 - fx) + at(x0 + 1, y0 + 1, a.oy) * fx) * fy
                dx += w * ox; dy += w * oy
            }
            out[k * 2] = u - dx
            out[k * 2 + 1] = v + dy
        }
        m.tex.rewind()
        return Mesh(m.verts, fb(out), m.indices, m.indexCount)
    }

    /** Sphere segment mesh (§2): a large-radius sphere (fixed 50 m — tens of
     *  metres, so the inter-ocular offset is a negligible fraction of the
     *  radius and stereo parallax artifacts are absent by construction).
     *  Density follows panoQuality: 60×48 ("vertex") or 72×60 ("vertexhq").
     *  U maps yaw linearly across the span, V maps pitch linearly; pole
     *  rows are duplicated-vertex rings stopping just short of the poles
     *  (no collapsed singularity point), leaving an invisible pinhole, so
     *  V clamps at the poles via CLAMP_TO_EDGE. The yaw seam duplicates
     *  vertices carrying U 0 and 1 so filtering never blends across the
     *  cut. The frame's full width maps across the span's yaw range, so
     *  intermediate spans are centred sub-windows clamping at the content
     *  edge. */
    private fun sphereSegment(deg: Float): Mesh {
        val hq = panoQuality == "vertexhq"
        val rows = if (hq) 72 else 60
        val cols = if (hq) 60 else 48
        val r = 50f
        val yawMax = Math.toRadians((deg / 2).toDouble())
        // Stop just short of the poles so the pole rows stay true rings of
        // distinct vertices instead of collapsing onto one point.
        val pitchMax = Math.PI / 2 - Math.toRadians(0.05)
        val verts = mutableListOf<Float>(); val texs = mutableListOf<Float>()
        for (iy in 0..rows) {
            val v = iy.toFloat() / rows
            // Linear pitch mapping (edge stretch removed: the Cardboard
            // pre-warp now owns all edge geometry). Centre v=0.5 = pitch 0.
            val pitch = (v - 0.5f) * 2f * pitchMax
            for (ix in 0..cols) {
                val u = ix.toFloat() / cols
                val yaw = -yawMax + u * 2 * yawMax
                val x = (r * Math.cos(pitch) * Math.sin(yaw)).toFloat()
                val y = (r * Math.sin(pitch)).toFloat()
                val z = (-r * Math.cos(pitch) * Math.cos(yaw)).toFloat()
                verts += listOf(x, y, z)
                // V in displayed-image space (v up): dome bottom samples the
                // frame bottom. The decoder's own flip/rotation lives in
                // uTexMat (§4) — baking a flip here too would mirror the
                // picture top-to-bottom.
                texs += listOf(u, v)
            }
        }
        val idx = mutableListOf<Short>()
        for (iy in 0 until rows) for (ix in 0 until cols) {
            val a = (iy * (cols + 1) + ix).toShort()
            val b = (a + 1).toShort(); val cc = ((iy + 1) * (cols + 1) + ix).toShort(); val d = (cc + 1).toShort()
            idx += listOf(a, cc, b, b, cc, d)
        }
        return Mesh(fb(verts.toFloatArray()), fb(texs.toFloatArray()), sb(idx.toShortArray()), idx.size)
    }

    private fun fb(a: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); position(0) }
    private fun sb(a: ShortArray): java.nio.ShortBuffer =
        ByteBuffer.allocateDirect(a.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer().apply { put(a); position(0) }

    private fun buildProgram(vs: String, fs: String): Int {
        fun compile(type: Int, src: String, tag: String): Int {
            val s = GLES20.glCreateShader(type)
            GLES20.glShaderSource(s, src)
            GLES20.glCompileShader(s)
            val ok = IntArray(1)
            GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) {
                val log = GLES20.glGetShaderInfoLog(s) ?: "?"
                android.util.Log.e("LimpetVR-GL", "$tag compile FAILED: $log")
                try { FileLog.e("LimpetVR-GL", "$tag compile FAILED: $log") } catch (_: Throwable) {}
            }
            return s
        }
        val v = compile(GLES20.GL_VERTEX_SHADER, vs, "VERT")
        val f = compile(GLES20.GL_FRAGMENT_SHADER, fs, "FRAG")
        return GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, v); GLES20.glAttachShader(it, f); GLES20.glLinkProgram(it)
            val ok = IntArray(1)
            GLES20.glGetProgramiv(it, GLES20.GL_LINK_STATUS, ok, 0)
            if (ok[0] == 0) {
                val log = GLES20.glGetProgramInfoLog(it) ?: "?"
                android.util.Log.e("LimpetVR-GL", "link FAILED: $log")
                try { FileLog.e("LimpetVR-GL", "link FAILED: $log") } catch (_: Throwable) {}
            }
            GLES20.glDeleteShader(v); GLES20.glDeleteShader(f)
        }
    }
}
