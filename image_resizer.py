from PIL import Image
import os

folder_dir = "dataset/test/3_esophagitis"
output_dir = "dataset/tes/esophagitis"
os.makedirs(output_dir, exist_ok=True)
resize_size = (720, 576)

count = 40
for images in os.listdir(folder_dir):
    if images.lower().endswith((".png", ".jpg", ".jpeg")):
        name, _ = os.path.splitext(images)
        print(name)
        count += 1

        path = os.path.join(folder_dir, images)
        image = Image.open(path).convert("RGB")  # Convert to RGB for consistency
        image = image.resize(resize_size, Image.Resampling.LANCZOS)
        save_path = os.path.join(output_dir, f'{name}.jpg')
        image.save(save_path)
