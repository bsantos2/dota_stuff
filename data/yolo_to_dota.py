import os
import cv2

# Paths
labels_dir = "/teamspace/studios/this_studio/data/dota-mod/labels/train"
images_dir = "/teamspace/studios/this_studio/data/dota-mod/images/train"
output_dir = "/teamspace/studios/this_studio/data/dota-mod/dota_labels/train"

os.makedirs(output_dir, exist_ok=True)

# Load class names
class_names={
    0: "plane", 
    1: "ship", 
    2: "storage tank", 
    3: "baseball diamond", 
    4: "tennis court", 
    5: "basketball court", 
    6: "ground track field", 
    7: "harbor", 
    8: "bridge", 
    9: "large vehicle", 
    10: "small vehicle", 
    11: "helicopter", 
    12: "roundabout", 
    13: "soccer ball field", 
    14: "swimming pool",
    15: "container crane"}

# Threshold to mark small objects as difficult
DIFFICULTY_AREA_THRESHOLD = 1000  # pixels

def convert_yolo_to_dota(yolo_label_path, image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return
    h, w = image.shape[:2]

    with open(yolo_label_path, "r") as f:
        lines = f.readlines()

    with open(output_path, "w") as out_f:
        for line in lines:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.strip().split())
            class_id = int(class_id)
            class_name = class_names[class_id]

            x1 = x1 * w
            x2 = x2 * w
            x3 = x3 * w
            x4 = x4 * w

            y1 = y1 * h
            y2 = y2 * h
            y3 = y3 * h
            y4 = y4 * h

            x_coords = [x1, x2, x3, x4]
            y_coords = [y1, y2, y3, y4]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            bw = x_max - x_min  # Bounding box width
            bh = y_max - y_min  # Bounding box height

            # Estimate area for difficulty
            area = bw * bh
            difficulty = 1 if area < DIFFICULTY_AREA_THRESHOLD else 0

            # Write DOTA-format line
            out_f.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f} {class_name} {difficulty}\n")

# Process all labels
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    base = os.path.splitext(label_file)[0]
    label_path = os.path.join(labels_dir, label_file)
    image_path = os.path.join(images_dir, base + ".jpg")  # Change to .jpg if needed
    output_path = os.path.join(output_dir, base + ".txt")

    convert_yolo_to_dota(label_path, image_path, output_path)

print("✅ Conversion complete. Check 'dota_labels/' for output.")
