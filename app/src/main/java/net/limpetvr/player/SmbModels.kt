/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import java.util.UUID

data class SmbConnection(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val host: String,
    val share: String,
    val domain: String = "",
    val username: String = "",
    // Stored encrypted via ConnectionStore (EncryptedSharedPreferences). Never logged.
    val password: String = ""
) {
    val label: String get() = if (name.isBlank()) "$host/$share" else name
    val unc: String get() = "smb://$host/$share"
}

data class SmbEntry(
    val name: String,
    val path: String, // backslash-separated path relative to share root, "" = root
    val isDir: Boolean,
    val size: Long
)

fun SmbEntry.isVideo(): Boolean {
    if (isDir) return false
    val n = name.lowercase()
    return n.endsWith(".mp4") || n.endsWith(".mkv") || n.endsWith(".avi") ||
        n.endsWith(".mov") || n.endsWith(".webm") || n.endsWith(".m4v") ||
        n.endsWith(".ts") || n.endsWith(".mpg") || n.endsWith(".mpeg")
}
