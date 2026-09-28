/* LimpetVR Player — GPL-3.0-only.
 * See LICENSE in the repository root.
 */
package net.limpetvr.player

enum class Projection(val label: String, val degrees: Float) {
    FLAT("Flat 2D", 0f),
    DEG180("180°", 180f),
    DEG220("220°", 220f),
    DEG270("270°", 270f),
    DEG360("360°", 360f),
    FISHEYE("Fisheye", 180f);
}

enum class Stereo(val label: String) {
    MONO("Mono / 2D"),
    SBS("Side-by-side"),
    TB("Top-bottom");
}
