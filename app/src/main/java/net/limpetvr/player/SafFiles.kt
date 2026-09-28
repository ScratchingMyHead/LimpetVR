/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.storage.StorageManager
import android.provider.DocumentsContract
import androidx.documentfile.provider.DocumentFile
import java.io.File

/** SD-card trees via the Storage Access Framework. Raw File access to
 *  shared SD locations is blocked on modern Android, so arbitrary SD
 *  browsing goes through a user-granted document tree (grants are kept
 *  in process memory only, never on disk). ExoPlayer plays the
 *  document Uris directly — no proxy involved. */
object SafFiles {
    data class SafEntry(
        val name: String,
        val uri: Uri,
        val isDir: Boolean,
        val size: Long,
        val mime: String?
    )

    fun dirAt(ctx: Context, treeUri: Uri, relPath: String): DocumentFile? {
        var dir = DocumentFile.fromTreeUri(ctx, treeUri) ?: return null
        if (relPath.isNotEmpty()) {
            for (seg in relPath.split('/')) {
                if (seg.isEmpty()) continue
                dir = dir.findFile(seg) ?: return null
                if (!dir.isDirectory) return null
            }
        }
        return dir
    }

    fun list(ctx: Context, treeUri: Uri, relPath: String): List<SafEntry> {
        val dir = try { dirAt(ctx, treeUri, relPath) } catch (_: Exception) { null }
            ?: return emptyList()
        return try {
            dir.listFiles().mapNotNull { d ->
                val n = d.name ?: return@mapNotNull null
                if (n.startsWith(".")) null
                else SafEntry(n, d.uri, d.isDirectory, d.length(), d.type)
            }.sortedWith(compareBy({ !it.isDir }, { it.name.lowercase() }))
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun isVideo(e: SafEntry): Boolean {
        if (e.isDir) return false
        if ((e.mime ?: "").startsWith("video/")) return true
        return LocalFiles.isVideoFile(File(e.name))
    }

    data class VolumeInfo(val uuid: String?, val desc: String)

    /** Removable (SD card) volumes for the roots list. */
    fun removableVolumes(ctx: Context): List<VolumeInfo> {
        return try {
            val sm = ctx.getSystemService(Context.STORAGE_SERVICE) as StorageManager
            sm.storageVolumes.mapNotNull { v ->
                try {
                    if (v.isRemovable) VolumeInfo(v.uuid, v.getDescription(ctx) ?: "SD card") else null
                } catch (_: Exception) { null }
            }
        } catch (_: Exception) { emptyList() }
    }

    /** Persisted tree grant covering a volume uuid (matches XXXX-XXXX prefix), if any. */
    fun grantedTree(ctx: Context, uuid: String?): String? {
        val persisted = try {
            ctx.contentResolver.persistedUriPermissions.map { it.uri.toString() }.toSet()
        } catch (_: Exception) { return null }
        for (u in persisted) {
            try {
                val docId = DocumentsContract.getTreeDocumentId(Uri.parse(u)) ?: continue
                val vol = docId.substringBefore(':', "")
                if (uuid != null && vol.equals(uuid, ignoreCase = true)) return u
            } catch (_: Exception) { }
        }
        return null
    }

    /** Take a persistable read grant for a picked tree. */
    fun takeGrant(ctx: Context, treeUri: Uri) {
        try {
            ctx.contentResolver.takePersistableUriPermission(treeUri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (_: Exception) { }
    }
}
