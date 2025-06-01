from PIL import Image
import os

folder_dir = "/home/bapary/Work/Local-Attention-Grouping-for-GA-and-IM-classification/dataset/end_val/3_esophagitis"
output_dir = "/home/bapary/Work/Local-Attention-Grouping-for-GA-and-IM-classification/dataset/end_val/resized/esophagitis"
resize_size = (720, 576)

count = 40
for images in os.listdir(folder_dir):
    if images.lower().endswith((".png", ".jpg", ".jpeg")):
        name, _ = os.path.splitext(images)
        print(name)
        count += 1

        path = os.path.join(folder_dir, images)
        image = Image.open(path).convert("RGB")  # Convert to RGB for consistency
        image = image.resize(resize_size, Image.ANTIALIAS)  # Resize image

        save_path = os.path.join(output_dir, f'{name}.jpg')
        image.save(save_path)
