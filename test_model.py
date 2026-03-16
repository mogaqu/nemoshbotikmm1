impot json
from ultralytics import YOLO
import numpy as np
import sys

# Load config
with open("MM2_Bot_Package/config.json", "r") as f:
    config = json.load(f)

output = []
output.append("Config weights: " + config['weights']['candy'])
output.append("Target classes: " + str(config.get('target_classes', {})))

# Load model
model = YOLO("MM2_Bot_Package/" + config['weights']['candy'])
output.append("Model classes: " + str(model.names))

# Test detection with dummy image
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
results = model(dummy_img, imgsz=320, conf=0.1)

output.append("Detection test completed!")

# Check if both classes are in the model
target_classes = config.get('target_classes', {'ball': 0})
output.append("Configured target classes: " + str(target_classes))
output.append("Model supports classes: " + str(model.names))

# Verify class IDs match
for name, cls_id in target_classes.items():
    if cls_id in model.names:
        output.append(f"  OK {name} (class {cls_id}): {model.names[cls_id]}")
    else:
        output.append(f"  FAIL {name} (class {cls_id}): NOT FOUND")

# Write to file
with open("test_output.txt", "w") as f:
    f.write("\n".join(output))