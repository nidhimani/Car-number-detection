import os
import json
import cv2 as cv
import ultralytics
import cv2


# from roboflow import Roboflow
# rf = Roboflow(api_key="iNsqDOkc7xvm9CuNGJv7")
# project = rf.workspace("project-epimx").project("guvi-task")
# version = project.version(2)
# dataset = version.download("yolov8")

model = ultralytics.YOLO("yolov8n.pt")

model.train(data = "guvi-task-2\data.yaml", epochs = 11)

model = ultralytics.YOLO(r"runs\detect\train\weights\last.pt")
# Directory paths
test_images_dir = 'test'  # Replace with the path to your test images
output_dir = 'recognised_images'  # Replace with the path where you want to save the cropped images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Run predictions on test images and save cropped bounding box regions
for image_name in os.listdir(test_images_dir):
    image_path = os.path.join(test_images_dir, image_name)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Run inference
    results = model(image_path)
    
    # Get bounding boxes from the results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Format: [xmin, ymin, xmax, ymax, confidence, class]
    
    # Process each bounding box
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box[:4])
        
        # Crop the region within the bounding box
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        # Create a unique name for each cropped image
        cropped_image_name = f"{os.path.splitext(image_name)[0]}_{i}.jpg"
        cropped_image_path = os.path.join(output_dir, cropped_image_name)
        
        # Save the cropped image
        cv2.imwrite(cropped_image_path, cropped_image)



