/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import androidx.core.content.ContextCompat
import java.io.File

/** Browsing files on the phone itself (the "This device" top-level entry). */
object LocalFiles {
    data class LocalEntry(val name: String, val file: File, val isDir: Boolean, val size: Long)

    fun needsPermission(ctx: Context): Boolean {
        return if (Build.VERSION.SDK_INT >= 33) {
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.READ_MEDIA_VIDEO) != PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        }
    }

    fun requestPermission(): String {
        return if (Build.VERSION.SDK_INT >= 33) Manifest.permission.READ_MEDIA_VIDEO
        else Manifest.permission.READ_EXTERNAL_STORAGE
    }

    fun externalRoot(): File = Environment.getExternalStorageDirectory()

    data class VolumeRoot(val label: String, val dir: File, val isPrimary: Boolean)

    /** Volume roots for the roots list: internal + removable SD. SD roots
     *  resolve via getExternalFilesDirs (strip the app-sandbox suffix) —
     *  no hidden APIs. Direct-File SD access needs All-files on API 30+. */
    fun volumeRoots(ctx: Context): List<VolumeRoot> {
        val roots = mutableListOf(VolumeRoot("Internal Storage", externalRoot(), true))
        try {
            val suffix = "/Android/data/${ctx.packageName}/files"
            for (f in ctx.getExternalFilesDirs(null)) {
                val p = f?.absolutePath ?: continue
                if (!p.endsWith(suffix)) continue
                val root = File(p.removeSuffix(suffix))
                if (roots.any { it.dir.absolutePath == root.absolutePath }) continue
                val tag = root.name.ifEmpty { "SD" }
                roots += VolumeRoot("SD card ($tag)", root, false)
            }
        } catch (_: Exception) {}
        return roots
    }

    /** True on API 30+ when direct SD File access is unavailable. Below 30
     *  the READ_* media permissions already cover shared storage. */
    fun needsFullAccess(): Boolean =
        Build.VERSION.SDK_INT >= 30 && !Environment.isExternalStorageManager()

    /** True when dir lives on an SD root that direct File cannot open yet. */
    fun sdBlocked(ctx: Context, dir: File): Boolean {
        if (!needsFullAccess()) return false
        val ap = dir.absolutePath
        return volumeRoots(ctx).any { !it.isPrimary &&
            (ap == it.dir.absolutePath || ap.startsWith(it.dir.absolutePath + "/")) }
    }

    /** (volume label, /relative/path) for a dir under a known root. */
    fun rootTitle(ctx: Context, dir: File): Pair<String, String> {
        val ap = dir.absolutePath
        for (vr in volumeRoots(ctx)) {
            val rp = vr.dir.absolutePath
            if (ap == rp || ap.startsWith(rp + "/"))
                return vr.label to ap.removePrefix(rp).ifEmpty { "/" }
        }
        return "Files" to ap
    }

    /** Send the user to the All-files toggle (deep-linked to this app);
     *  callers re-check on return (no system popup exists for this grant). */
    fun requestFullAccess(activity: android.app.Activity) {
        try {
            activity.startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:${activity.packageName}")))
        } catch (_: Exception) {
            try { activity.startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
            catch (_: Exception) {}
        }
    }

    fun isVideoFile(f: File): Boolean {
        if (!f.isFile) return false
        val n = f.name.lowercase()
        return n.endsWith(".mp4") || n.endsWith(".mkv") || n.endsWith(".avi") ||
            n.endsWith(".mov") || n.endsWith(".webm") || n.endsWith(".m4v") ||
            n.endsWith(".ts") || n.endsWith(".mpg") || n.endsWith(".mpeg")
    }

    /** List a directory: subdirs first, then playable videos. Hidden dirs skipped. */
    fun list(dir: File): List<LocalEntry> {
        val kids = dir.listFiles() ?: return emptyList()
        return kids
            .filter { !it.name.startsWith(".") && (it.isDirectory || isVideoFile(it)) }
            .map { LocalEntry(it.name, it, it.isDirectory, if (it.isFile) it.length() else 0L) }
            .sortedWith(compareBy({ !it.isDir }, { it.name.lowercase() }))
    }

    fun toSmbEntry(e: LocalEntry): SmbEntry =
        SmbEntry(e.name, e.file.absolutePath, e.isDir, e.size)
}
