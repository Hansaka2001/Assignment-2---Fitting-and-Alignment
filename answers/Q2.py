import cv2
import numpy as np
import matplotlib.pyplot as plt

focal_length_mm = 8.0
distance_z_mm = 720.0
pixel_size_mm = 0.0022


magnification = distance_z_mm / focal_length_mm
mm_per_pixel = magnification * pixel_size_mm

print(f"Camera Scale: 1 pixel = {mm_per_pixel:.4f} mm")

img_path = 'earrings.jpg' 
image = cv2.imread(img_path)

if image is None:
    print(f"Error: Could not load image at '{img_path}'. Please check the file name and path.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    output_img = image.copy()
    
    print("\n Measurement Results ")
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        real_width = w * mm_per_pixel
        real_height = h * mm_per_pixel
        
        print(f"Earring {i+1}: Width = {real_width:.2f} mm, Height = {real_height:.2f} mm")
        
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cx, cy = x + w//2, y + h//2
        cv2.drawMarker(output_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        text_label = f"W:{real_width:.1f}mm H:{real_height:.1f}mm"
        
        (text_w, text_h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_img, (x, y - text_h - 10), (x + text_w, y), (0, 0, 0), -1)
        cv2.putText(output_img, text_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(output_rgb)
    plt.title("Automated Earring Measurement (Question 2)", fontsize=16, fontweight='bold', pad=15)
    plt.axis('off') 
    
    plt.tight_layout()
    plt.savefig('q2_earring_measurement.png', dpi=300, bbox_inches='tight')
    print("\nSaved final image as 'q2_earring_measurement.png'")
    plt.show()