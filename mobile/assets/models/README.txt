Place the converted model here (and in the platform locations noted below).

Android: android/app/src/main/assets/yolo26n.tflite
iOS:     add yolo26n.mlpackage to the Runner target in Xcode

Convert from the desktop project:
  yolo export model=yolo26n.pt format=tflite imgsz=640     (Android)
  yolo export model=yolo26n.pt format=coreml imgsz=640     (iOS, on macOS)

For a custom-trained model, export models/custom_<project>.pt the same way
and update `modelPath` in lib/main.dart.
