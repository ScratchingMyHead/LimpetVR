/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import org.json.JSONArray
import org.json.JSONObject

/** Encrypted store for SMB connections (passwords never touch plain prefs). */
class ConnectionStore(ctx: Context) {
    private val prefs: SharedPreferences = try {
        val masterKey = MasterKey.Builder(ctx).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build()
        EncryptedSharedPreferences.create(
            ctx, "limpetvr_conns", masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    } catch (_: Exception) {
        ctx.getSharedPreferences("limpetvr_conns_fallback", Context.MODE_PRIVATE)
    }

    fun load(): MutableList<SmbConnection> {
        val raw = prefs.getString("connections", "[]") ?: "[]"
        val out = mutableListOf<SmbConnection>()
        val arr = JSONArray(raw)
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            out += SmbConnection(
                id = o.optString("id", java.util.UUID.randomUUID().toString()),
                name = o.optString("name", ""),
                host = o.optString("host", ""),
                share = o.optString("share", ""),
                domain = o.optString("domain", ""),
                username = o.optString("user", ""),
                password = o.optString("pass", "")
            )
        }
        return out
    }

    fun save(all: List<SmbConnection>) {
        val arr = JSONArray()
        for (c in all) {
            arr.put(JSONObject().apply {
                put("id", c.id); put("name", c.name); put("host", c.host)
                put("share", c.share); put("domain", c.domain)
                put("user", c.username); put("pass", c.password)
            })
        }
        prefs.edit().putString("connections", arr.toString()).apply()
    }
}

/** The process-wide current directory: connection id + path. Written by the
 *  2D list, the VR browser and every play (2D play, in-VR play, next/prev,
 *  file-manager launches), read by both surfaces whenever either is opened.
 *  Deliberately NOT persisted — a fresh cold start opens the top level. */
object SessionMemory {
    var lastConnectionId: String = ""
    var lastPath: String = ""
}

/** Non-secret UI prefs: display mode, last location, buffer. */
class SettingsStore(ctx: Context) {
    private val p = ctx.getSharedPreferences("limpetvr_settings", Context.MODE_PRIVATE)

    var projection: Projection
        get() = runCatching { Projection.valueOf(p.getString("projection", "DEG180")!!) }.getOrDefault(Projection.DEG180)
        set(v) { p.edit().putString("projection", v.name).apply() }
    var stereo: Stereo
        get() = runCatching { Stereo.valueOf(p.getString("stereo", "SBS")!!) }.getOrDefault(Stereo.SBS)
        set(v) { p.edit().putString("stereo", v.name).apply() }
    var startInVrBrowser: Boolean
        get() = p.getBoolean("vr_browser", true)
        set(v) { p.edit().putBoolean("vr_browser", v).apply() }
    var bufferKb: Int
        get() = p.getInt("buffer_kb", 256).coerceIn(32, 2048)
        set(v) { p.edit().putInt("buffer_kb", v).apply() }
    // --- optics / comfort ---
    /** Scales the SDK's per-eye field of view (0.5..1.5, 1 = the viewer
     *  profile's own angles). Baseline forces 1. */
    var fovScale: Float
        get() = p.getFloat("fov_scale", 1f).coerceIn(0.5f, 1.5f)
        set(v) { p.edit().putFloat("fov_scale", v).apply() }
    /** Flat screen size multiplier, 0.5-10 (1 = default). */
    var screenSize: Float
        get() = p.getFloat("screen_size", 1f).coerceIn(0.5f, 10f)
        set(v) { p.edit().putFloat("screen_size", v).apply() }
    /** Flat screen curvature, 0-1 (0 = the flat plane, 1 = a spherical cap
     *  bent around the viewer; IMAX-style wrap). Baseline forces 0. */
    var screenCurve: Float
        get() = p.getFloat("screen_curve", 0f).coerceIn(0f, 1f)
        set(v) { p.edit().putFloat("screen_curve", v).apply() }
    /** Panorama mesh density: "vertex" (60x48) or "vertexhq" (72x60). */
    var panoQuality: String
        get() = p.getString("pano_quality", "vertex") ?: "vertex"
        set(v) { p.edit().putString("pano_quality", v).apply() }
    /** Skip the SDK's Cardboard lens-distortion pass (raw stereo). */
    var disableDist: Boolean
        get() = p.getBoolean("disable_dist", false)
        set(v) { p.edit().putBoolean("disable_dist", v).apply() }
    /** Show the gaze tooltip pill above the reticle. */
    var enableTooltip: Boolean
        get() = p.getBoolean("enable_tooltip", true)
        set(v) { p.edit().putBoolean("enable_tooltip", v).apply() }
    /** Full interpupillary distance in mm. */
    var ipdMm: Float
        get() = p.getFloat("ipd", 64f).coerceIn(40f, 80f)
        set(v) { p.edit().putFloat("ipd", v).apply() }
    /** Swap left/right eye images (some viewers + phones need it). */
    var swapEyes: Boolean
        get() = p.getBoolean("swap_eyes", false)
        set(v) { p.edit().putBoolean("swap_eyes", v).apply() }
    /** Browser panel distance in meters. */
    var panelDistM: Float
        get() = p.getFloat("panel_d", 2.4f).coerceIn(1.2f, 5f)
        set(v) { p.edit().putFloat("panel_d", v).apply() }
    /** Video scale / zoom number z: domes slide the sphere toward the viewer
     *  (§5), the FLAT quad minifies its texture window. Range 0.1-20
     *  both directions; baseline forces 1. */
    var videoZoom: Float
        get() = p.getFloat("zoom", 1f).coerceIn(0.1f, 20f)
        set(v) { p.edit().putFloat("zoom", v).apply() }
    /** Gaze dwell-to-select in ms. */
    var dwellMs: Long
        get() = p.getLong("dwell", 1500L).coerceIn(400L, 4000L)
        set(v) { p.edit().putLong("dwell", v).apply() }
    /** Rewind/fast-forward jump in seconds (2D setting, VR buttons use it). */
    var skipSecs: Int
        get() = p.getInt("skip_secs", 10).coerceIn(2, 60)
        set(v) { p.edit().putInt("skip_secs", v).apply() }
    /** Shaping grid enabled (per shape id, see shaping/shapes.json). */
    fun shapeEnabled(id: String): Boolean = p.getBoolean("shape_on_$id", false)
    fun setShapeEnabled(id: String, v: Boolean) { p.edit().putBoolean("shape_on_$id", v).apply() }
    /** Shaping grid weight 0..100% (per shape id). Default 100%. */
    fun shapeWeight(id: String): Float = p.getFloat("shape_w_$id", 100f).coerceIn(0f, 100f)
    fun setShapeWeight(id: String, v: Float) { p.edit().putFloat("shape_w_$id", v).apply() }
    /** Granted SAF tree URIs (SD cards). Grants themselves persist via the
     *  ContentResolver; this just remembers which trees were picked. */
    var safTrees: Set<String>
        get() = p.getStringSet("saf_trees", emptySet())?.toSet() ?: emptySet()
        set(v) { p.edit().putStringSet("saf_trees", v.toSet()).apply() }

    /** Pin video in front of the viewer (screen lock); off = look-around. */
    var pinVideo: Boolean
        get() = p.getBoolean("pin_video", false)
        set(v) { p.edit().putBoolean("pin_video", v).apply() }
    /** Diagnostics: drive head tracking with a scripted sweep instead of sensors. */
    var testSweep: Boolean
        get() = p.getBoolean("test_sweep", false)
        set(v) { p.edit().putBoolean("test_sweep", v).apply() }
    /** Play video without audio (isolates A/V clock/sync involvement). */
    var muteAudioTrack: Boolean
        get() = p.getBoolean("mute_audio_track", false)
        set(v) { p.edit().putBoolean("mute_audio_track", v).apply() }
    /** Prefer software video decoders (compat for streams that stall HW decoders). */
    var softwareDecode: Boolean
        get() = p.getBoolean("sw_decode", false)
        set(v) { p.edit().putBoolean("sw_decode", v).apply() }
    /** Last non-fisheye screen shape (the Lens toggle flips between this
     *  and FISHEYE without losing the dome setting). */
    var screenProj: Projection
        get() = runCatching { Projection.valueOf(p.getString("screen_proj", "DEG180")!!) }.getOrDefault(Projection.DEG180)
        set(v) { p.edit().putString("screen_proj", v.name).apply() }
    /** Cardboard lens distortion k1 (0..1, standard coefficients, 0.34 default). */
    var lensK1: Float
        get() = p.getFloat("lens_k1", 0.34f).coerceIn(0f, 1f)
        set(v) { p.edit().putFloat("lens_k1", v).apply() }
    /** Cardboard lens distortion k2 (0..1, standard coefficients, 0.55 default). */
    var lensK2: Float
        get() = p.getFloat("lens_k2", 0.55f).coerceIn(0f, 1f)
        set(v) { p.edit().putFloat("lens_k2", v).apply() }
    /** Master strength for the whole pre-warp (1 = physical, 0 = off).
     *  Absorbs viewer profile error, eye-to-screen error and xdpi skew
     *  in one knob: f = 1 + S*(K1*r^2 + K2*r^4). */
    var lensStrength: Float
        get() = p.getFloat("lens_strength", 1f).coerceIn(0f, 3f)
        set(v) { p.edit().putFloat("lens_strength", v).apply() }
    /** Fisheye circle radius multiplier (1 = spec default: quarter frame
     *  width). Calibration per camera rig. */
    var fisheyeRadius: Float
        get() = p.getFloat("fish_r", 1f).coerceIn(0.5f, 1.5f)
        set(v) { p.edit().putFloat("fish_r", v).apply() }
    /** Mirror the right eye's fisheye circle lookup (some rigs mirror it). */
    var fisheyeMirrorR: Boolean
        get() = p.getBoolean("fish_mirror_r", false)
        set(v) { p.edit().putBoolean("fish_mirror_r", v).apply() }
    /** Convergence trim: uniform clip-space offset per eye (±0.15).
     *  Positive converges the halves. Baseline forces -0.040;
     *  live-tunable after. */
    var convTrim: Float
        get() = p.getFloat("conv_trim", -0.04f).coerceIn(-0.15f, 0.15f)
        set(v) { p.edit().putFloat("conv_trim", v).apply() }
    // --- startup baseline (§10) ---
    /** Last baseline fire, epoch ms (0 = never). */
    var baselineAt: Long
        get() = p.getLong("baseline_at", 0L)
        set(v) { p.edit().putLong("baseline_at", v).apply() }
    /** Baseline has fired at least once (fresh-install detection). */
    var baselineEver: Boolean
        get() = p.getBoolean("baseline_ever", false)
        set(v) { p.edit().putBoolean("baseline_ever", v).apply() }

    /** Startup baseline (§10): forces the calibration table, persists
     *  eye separation / projection / stereo / menu timings and everything
     *  else unlisted by leaving it untouched. Records the fire time. */
    fun fireBaseline(nowMs: Long) {
        lensK1 = 0.34f; lensK2 = 0.55f
        videoZoom = 1f
        fovScale = 1f
        screenSize = 1f
        screenCurve = 0f
        panoQuality = "vertex"
        disableDist = false
        enableTooltip = true
        pinVideo = false; swapEyes = false; testSweep = false
        convTrim = -0.04f
        baselineAt = nowMs
        baselineEver = true
    }
    /** Re-apply the lens/calibration defaults only, leaving the user's
     *  comfort and quality settings alone.
     *
     *  The 30-minute expiry used to call fireBaseline(), which also reset
     *  screen size, curve, zoom, FOV and a dozen other things the user had
     *  chosen. Screen geometry is not a calibration result, so a stale
     *  session must not quietly undo it - otherwise the same "it reset
     *  itself" surprise returns after half an hour away from the app. */
    fun fireCalibrationBaseline(nowMs: Long) {
        lensK1 = 0.34f; lensK2 = 0.55f
        convTrim = -0.04f
        swapEyes = false
        baselineAt = nowMs
    }

    /** Look-up tilt (deg, 10..60) that opens the VR play menu when it is on top. */
    var menuAngleUp: Float
        get() = p.getFloat("menu_angle_up",
            kotlin.math.abs(p.getFloat("menu_angle", 40f)).coerceIn(10f, 60f)).coerceIn(10f, 60f)
        set(v) { p.edit().putFloat("menu_angle_up", v).apply() }
    /** Look-down tilt (deg, stored negative, -60..-10) that opens the menu at the bottom. */
    var menuAngleDown: Float
        get() = p.getFloat("menu_angle_down",
            -kotlin.math.abs(p.getFloat("menu_angle", 40f)).coerceIn(10f, 60f)).coerceIn(-60f, -10f)
        set(v) { p.edit().putFloat("menu_angle_down", v).apply() }
    /** Which side the play menu lives on; flipped by its ⇅ button. */
    var menuTop: Boolean
        get() = p.getBoolean("menu_top", true)
        set(v) { p.edit().putBoolean("menu_top", v).apply() }
    /** Head-circle gesture arms recenter aim (VIDEO mode, menu closed). */
    var circleRecenter: Boolean
        get() = p.getBoolean("circle_recenter", true)
        set(v) { p.edit().putBoolean("circle_recenter", v).apply() }
}
