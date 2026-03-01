import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

FOCAL_LENGTH = 8.0
OBJECT_Z_DIST = 720.0
PIXEL_PITCH = 0.0022

SCALE_MM_PER_PX = (PIXEL_PITCH * OBJECT_Z_DIST) / FOCAL_LENGTH

image_bgr = cv2.imread("../inputs/earrings.jpg")
if image_bgr is None:
    raise FileNotFoundError("Could not find 'earrings.jpg'")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

_, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

valid_shapes = {}
for idx, contour in enumerate(contours):
    if cv2.contourArea(contour) > 500:  
        parent_id = hierarchy[0][idx][3]
        valid_shapes[idx] = {
            'is_exterior': (parent_id == -1),
            'parent': parent_id,
            'rect': cv2.boundingRect(contour)
        }

annotated_img = image_rgb.copy()
table_rows = []
recorded_types = {"Exterior": False, "Interior": False}

for idx, data in valid_shapes.items():
    x, y, w, h = data['rect']
    
    if not data['is_exterior'] and data['parent'] in valid_shapes:
        parent_rect = valid_shapes[data['parent']]['rect']
        bottom_y = y + h
        y = parent_rect[1]  
        h = bottom_y - y    
        
    shape_type = "Exterior" if data['is_exterior'] else "Interior"
    
    color = (0, 255, 0) if data['is_exterior'] else (255, 165, 0)
    
    if not recorded_types[shape_type]:
        w_mm, h_mm = w * SCALE_MM_PER_PX, h * SCALE_MM_PER_PX
        table_rows.append([shape_type, w, f"{w_mm:.2f}", h, f"{h_mm:.2f}"])
        recorded_types[shape_type] = True

    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
    
    cx, cy = x + w // 2, y + h // 2
    
    cv2.line(annotated_img, (x, cy), (x + w, cy), color, 1)
    cv2.circle(annotated_img, (x, cy), 4, color, -1)
    cv2.circle(annotated_img, (x + w, cy), 4, color, -1)
    
    cv2.line(annotated_img, (cx, y), (cx, y + h), color, 1)
    cv2.circle(annotated_img, (cx, y), 4, color, -1)
    cv2.circle(annotated_img, (cx, y + h), 4, color, -1)

img_h, img_w = image_rgb.shape[:2]
max_mm_w = img_w * SCALE_MM_PER_PX
max_mm_h = img_h * SCALE_MM_PER_PX

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(6, 1) 
ax_img = fig.add_subplot(gs[:5, 0]) 
ax_tbl = fig.add_subplot(gs[5, 0])  

ax_img.imshow(annotated_img, extent=[0, max_mm_w, max_mm_h, 0])

ax_img.xaxis.set_major_locator(MultipleLocator(10))
ax_img.yaxis.set_major_locator(MultipleLocator(10))
ax_img.xaxis.set_minor_locator(MultipleLocator(2))
ax_img.yaxis.set_minor_locator(MultipleLocator(2))

ax_img.grid(which='major', color='#333333', linestyle='-', linewidth=1.2, alpha=0.7)
ax_img.grid(which='minor', color='#777777', linestyle='--', linewidth=0.6, alpha=0.4)

ax_img.set_xlabel("Physical Width (mm)", fontsize=11, fontweight='bold')
ax_img.set_ylabel("Physical Height (mm)", fontsize=11, fontweight='bold')
ax_img.set_title("Machine Vision: Earring Dimensional Analysis", fontsize=14, fontweight='bold', pad=15)

ax_tbl.axis('off')
headers = ["Boundary Type", "Pixel Width", "Width (mm)", "Pixel Height", "Height (mm)"]
data_table = ax_tbl.table(cellText=table_rows, colLabels=headers, loc='center', cellLoc='center')
data_table.scale(1, 1.8)
data_table.set_fontsize(11)

for (row_idx, col_idx), cell in data_table.get_celld().items():
    cell.set_edgecolor('#bdc3c7')
    if row_idx == 0:
        cell.set_facecolor('#2c3e50') 
        cell.set_text_props(color='white', weight='bold')
    else:
        cell.set_facecolor('#ecf0f1') 

plt.tight_layout()
plt.savefig("q2_final_measurements.png", dpi=300, bbox_inches='tight')
print("\nImage saved as 'q2_final_measurements.png'")
plt.show()