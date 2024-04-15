import cv2
import os

# Define the target size for resizing
target_width = 791
target_height = 548

# Directory containing your images
input_dir = "./ZeldaLevels/train/playable"
output_dir = "resizing/train/playable"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # adjust the extensions as per your images
        # Read the image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Resize the image to the target size
        resized_image = cv2.resize(image, (target_width, target_height))

        # Write the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_image)

print("All images resized and saved successfully!")
