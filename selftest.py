"""Quick non-GUI smoke test: verifies the engine imports, detects hardware,
loads the model, and runs detection on a synthetic image."""
import numpy as np
from detector import HardwareProfile, Detector

print("Detecting hardware...")
prof = HardwareProfile.detect()
print("  ", prof.summary)
print("   device =", prof.device, "| model =", prof.recommended_model,
      "| imgsz =", prof.imgsz)

print("Loading model (downloads on first run)...")
det = Detector(prof.recommended_model, prof)
print("   loaded:", det.weights, "| classes:", len(det.names))

print("Running detection on a synthetic 640x480 image...")
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
annotated, counts = det.predict_image(img, conf=0.25)
print("   output frame:", annotated.shape, "| counts:", counts)
print("OK - engine works.")
