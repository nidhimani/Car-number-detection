import cv2
import easyocr
import csv
import os

def ocr_with_easyocr(image_file):
    # Replace with your desired OCR logic using EasyOCR
    reader = easyocr.Reader(['en', 'ar'])  # Initialize EasyOCR with English language
    result = reader.readtext(image_file)  # Perform OCR on the image file
    return result

# Directory containing OCR results
output_directory = "recognised_images"

# List to store results
ocr_results = []
csv_file = 'ocr_results.csv'

# Iterate through files in the output directory
for file_name in sorted(os.listdir(output_directory)):
    image_file = os.path.join(output_directory, file_name)
    if os.path.isfile(image_file):
        # Perform OCR on the image file
        result = ocr_with_easyocr(image_file)
        extracted_text = ' '.join([text[1] for text in result])
        ocr_results.append((file_name, extracted_text))

        # Write to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['img_id', 'extracted_text'])
            # Write data rows
            writer.writerows(ocr_results)

