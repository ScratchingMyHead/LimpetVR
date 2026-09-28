pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Vendored Google VR SDK (sdk-base AAR) — see repo/com/google/vr/...
        maven { url = uri("repo") }
    }
}
rootProject.name = "LimpetVR"
include(":app")
