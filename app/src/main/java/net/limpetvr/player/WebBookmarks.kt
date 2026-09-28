/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** A saved web address. Bookmarks are entered on the 2D screen (URLs tab),
 *  because there is no keyboard in VR; the web panel just lists them. */
data class WebBookmark(
    val id: String,
    val title: String,
    val url: String
)

/** Plain (unencrypted) store: bookmarks are not secrets. */
class BookmarkStore(ctx: Context) {
    private val prefs = ctx.getSharedPreferences("limpetvr_bookmarks", Context.MODE_PRIVATE)

    fun load(): MutableList<WebBookmark> {
        val raw = prefs.getString("bookmarks", "[]") ?: "[]"
        val out = mutableListOf<WebBookmark>()
        val arr = JSONArray(raw)
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            out += WebBookmark(
                id = o.optString("id", java.util.UUID.randomUUID().toString()),
                title = o.optString("title", ""),
                url = o.optString("url", "")
            )
        }
        return out
    }

    fun save(all: List<WebBookmark>) {
        val arr = JSONArray()
        for (b in all) {
            arr.put(JSONObject().apply {
                put("id", b.id); put("title", b.title); put("url", b.url)
            })
        }
        prefs.edit().putString("bookmarks", arr.toString()).apply()
    }
}

/** A bare word typed in the URL bar means a search, not a host. */
fun normalizeUrl(raw: String): String {
    val s = raw.trim()
    if (s.isEmpty()) return ""
    if (s.startsWith("http://", true) || s.startsWith("https://", true)) return s
    if (s.contains("://")) return s
    // localhost and bare IPv4 stay URLs; dotted names with a known-ish TLD too
    val looksHost = s.startsWith("localhost", true) ||
        s.matches(Regex("^\\d{1,3}(\\.\\d{1,3}){3}(:\\d+)?(/.*)?$")) ||
        s.startsWith("192.168.") || s.startsWith("10.") || s.startsWith("127.")
    if (looksHost || s.contains(".") && !s.contains(" ")) return "https://$s"
    return "https://duckduckgo.com/?q=" + java.net.URLEncoder.encode(s, "UTF-8")
}
