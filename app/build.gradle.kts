plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Release signing comes from the (untracked, gitignored) local.properties:
// release.storeFile / release.storePassword / release.keyAlias /
// release.keyPassword. Generate once per machine with:
//   keytool -genkeypair -keystore <path> -alias limpetvr -keyalg RSA \
//     -keysize 2048 -validity 10950 -dname "CN=LimpetVR"
// and guard the keystore + local.properties (chmod 600). Losing the keystore
// means future releases can't update existing installs.
val localPropsFile = rootProject.file("local.properties")
if (localPropsFile.exists()) {
    localPropsFile.forEachLine { line ->
        val t = line.trim()
        if (t.isNotEmpty() && !t.startsWith("#") && "=" in t) {
            project.extra.set(t.substringBefore("="), t.substringAfter("="))
        }
    }
}

android {
    namespace = "net.limpetvr.player"
    compileSdk = 34

    signingConfigs {
        create("release") {
            storeFile = project.findProperty("release.storeFile")?.let { file(it as String) }
            storePassword = project.findProperty("release.storePassword") as String?
            keyAlias = project.findProperty("release.keyAlias") as String?
            keyPassword = project.findProperty("release.keyPassword") as String?
        }
    }

    defaultConfig {
        applicationId = "net.limpetvr.player"
        minSdk = 26
        targetSdk = 34
        versionCode = 60
        versionName = "0.6.32"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Signed only when local.properties carries the release key
            // (keystore itself lives outside git; see README Download).
            // Without it the release APK stays unsigned.
            if (project.hasProperty("release.storeFile")) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
        debug {
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
}

// Authored shape grids live in misc/shapes.json (edited via misc/shapemesh.py).
// Copy into APK assets at build time; runtime loads "shaping/shapes.json".
tasks.register<Copy>("copyShapingAssets") {
    from("$rootDir/misc/shapes.json")
    into("src/main/assets/shaping")
}
tasks.named("preBuild") { dependsOn("copyShapingAssets") }

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.coordinatorlayout:coordinatorlayout:1.2.0")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // Media3 ExoPlayer for playback (http from our local proxy)
    implementation("androidx.media3:media3-exoplayer:1.4.1")
    implementation("androidx.media3:media3-ui:1.4.1")
    implementation("androidx.media3:media3-common:1.4.1")

    // Google VR SDK: GvrView (stereo renderer + head tracking) and Cardboard
    // lens-distortion correction. Vendored under repo/ (offline, license-pinned).
    implementation("com.google.vr:sdk-base:1.200.0")

    // SMB2/3 client — directory listing + random-access reads
    implementation("com.hierynomus:smbj:0.14.0") {
        exclude(group = "org.slf4j", module = "slf4j-simple")
    }
    implementation("org.slf4j:slf4j-android:1.7.36")
}
