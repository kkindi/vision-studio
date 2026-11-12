// Vision Studio mobile - offline object detection & counting (Android + iOS).
//
// Mirrors the desktop app: live-camera YOLO detection with per-label counts and
// an adjustable confidence threshold. Runs fully on-device (TFLite on Android,
// CoreML on iOS) via the ultralytics_yolo plugin.
//
// NOTE: this compiles against the ultralytics_yolo plugin API; once the Flutter
// toolchain is installed we run `flutter pub get` + `flutter run` and reconcile
// any minor API differences against the installed plugin version.

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

void main() => runApp(const VisionStudioApp());

class VisionStudioApp extends StatelessWidget {
  const VisionStudioApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vision Studio',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const DetectionScreen(),
    );
  }
}

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  final YOLOViewController _controller = YOLOViewController();
  Map<String, int> _counts = {};
  double _confidence = 0.35;

  // Aggregate live detections into per-label counts.
  void _onResults(List<YOLOResult> results) {
    final counts = <String, int>{};
    for (final r in results) {
      counts[r.className] = (counts[r.className] ?? 0) + 1;
    }
    if (mounted) setState(() => _counts = counts);
  }

  @override
  Widget build(BuildContext context) {
    final total = _counts.values.fold<int>(0, (a, b) => a + b);
    final entries = _counts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Scaffold(
      appBar: AppBar(title: const Text('Vision Studio - Live Detection')),
      body: Stack(
        children: [
          // Live camera + on-device detection.
          YOLOView(
            modelPath: 'yolo26n', // bundled model asset (see assets/models)
            task: YOLOTask.detect,
            controller: _controller,
            onResult: _onResults,
          ),

          // Counts overlay (top-left), like the desktop "Counts" panel.
          Positioned(
            left: 12,
            top: 12,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.55),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text('Total in view: $total',
                      style: const TextStyle(
                          fontSize: 18, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 6),
                  if (entries.isEmpty)
                    const Text('Point the camera at objects...')
                  else
                    ...entries.map((e) => Padding(
                          padding: const EdgeInsets.only(top: 2),
                          child: Text('${e.key}:  ${e.value}'),
                        )),
                ],
              ),
            ),
          ),
        ],
      ),

      // Confidence slider, like the desktop control.
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
        child: Row(
          children: [
            const Text('Confidence'),
            Expanded(
              child: Slider(
                value: _confidence,
                min: 0.10,
                max: 0.90,
                divisions: 16,
                label: _confidence.toStringAsFixed(2),
                onChanged: (v) {
                  setState(() => _confidence = v);
                  _controller.setConfidenceThreshold(v);
                },
              ),
            ),
            Text(_confidence.toStringAsFixed(2)),
          ],
        ),
      ),
    );
  }
}
