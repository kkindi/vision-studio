# Vision Studio Mobile (Flutter — Android + iOS)

Offline object detection & counting on phones, sharing the same YOLO model as the
desktop app. One Flutter codebase builds both Android and iOS.

> Status: **experimental scaffold — it has never been built or run.** The
> `lib/main.dart` here is a starting point and will likely need changes to match
> the current `ultralytics_yolo` plugin API. Treat this folder as a plan, not a
> working app; the desktop app in the parent folder is the tested product.

---

## 1. Install the toolchain (one time)

**Android (buildable from Windows):**
1. Install **JDK 17** (older Java versions are too old for current Android builds):
   `winget install Microsoft.OpenJDK.17`
2. Install **Flutter SDK**: download from https://docs.flutter.dev/get-started/install/windows
   and add `flutter\bin` to PATH (or `git clone https://github.com/flutter/flutter`).
3. Install **Android SDK** — easiest via **Android Studio**
   (`winget install Google.AndroidStudio`), which bundles the SDK, an emulator,
   and the Flutter/Dart plugins. Then:
   ```bat
   flutter doctor --android-licenses   REM accept all
   flutter doctor                      REM should show Android toolchain OK
   ```

**iOS (needs macOS):** on a Mac, install Xcode, then
`sudo gem install cocoapods`. iOS apps can only be built/signed from macOS.

## 2. Generate the platform projects

This folder holds `lib/` + `pubspec.yaml`. Generate the native Android/iOS
wrappers in place:

```bat
cd mobile
flutter create . --org com.visionstudio --project-name visionstudio_mobile
flutter pub get
```

(If `ultralytics_yolo` fails to resolve, run `flutter pub add ultralytics_yolo`
to pull the current version, then re-run `flutter pub get`.)

## 3. Convert the model to the mobile formats

The plugin needs **TFLite** (Android) and **CoreML** (iOS). From the desktop
project's venv, or in a free Google Colab notebook (fastest, no local setup):

```bash
# Android
yolo export model=yolo26n.pt format=tflite imgsz=640
# iOS (run on macOS — coremltools works best there)
yolo export model=yolo26n.pt format=coreml imgsz=640 nms=True
```

Place the outputs:
- Android: `android/app/src/main/assets/yolo26n.tflite`
- iOS: add `yolo26n.mlpackage` to the Runner target in Xcode

(For a **custom-trained** model, export `runs/<project>/weights/best.pt` the same
way and update `modelPath` in `lib/main.dart`.)

## 4. Run

```bat
flutter devices                 REM list emulators / plugged-in phones
flutter run                     REM build + install + launch
flutter build apk --release     REM standalone APK to sideload
```

Grant the camera permission when prompted. You'll see live detection with the
per-label counts overlay and the confidence slider — the phone version of the
desktop app.

## Feature parity plan

- [x] Live camera detection + per-label counts (this scaffold)
- [x] Confidence control
- [ ] Image / video file modes (gallery picker)
- [ ] Unique counting via tracking (plugin tracking API)
- [ ] Custom-model picker in-app
- [ ] Auto hardware optimization (the plugin already uses GPU/NNAPI/CoreML where available)
