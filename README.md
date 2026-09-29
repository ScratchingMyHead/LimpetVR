# LimpetVR

Native Cardboard VR web browser and video player for Android with in-headset SMB (SMB2/3)
browsing and direct streaming.

- Phone + Cardboard stereo rendering
- Browse servers, folders, and files without leaving VR
- Projections: Flat 2D/imax, SBS/TB, 180/220/270/360 domes, fisheye

This is a work in progress. The web browser is not fully functional.

## Download

Current release APK
[LimpetVR-0.6.32.apk](https://github.com/ScratchingMyHead/LimpetVR/releases/download/v0.6.32/LimpetVR-0.6.32.apk).
All releases: [releases page](https://github.com/ScratchingMyHead/LimpetVR/releases).

## Sensors

Head tracking needs a gyroscope (same as Cardboard itself).

## Build

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Requires Android SDK 34, Kotlin 1.9.x.

## License

GPL-3.0-only — see [LICENSE](LICENSE). Anyone distributing a product
built from this code must provide its full source under the same terms.
