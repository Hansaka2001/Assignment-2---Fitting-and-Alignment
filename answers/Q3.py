import cv2
import numpy as np

selected_corners = []


def capture_clicks(event, x, y, flags, param):
    global selected_corners, ui_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_corners) < 4:
            selected_corners.append((x, y))
            cv2.circle(ui_image, (x, y), 6, (0, 255, 0), -1)

            if len(selected_corners) > 1:
                cv2.line(
                    ui_image, selected_corners[-2], selected_corners[-1], (0, 255, 255), 2)

            if len(selected_corners) == 4:
                cv2.line(
                    ui_image, selected_corners[-1], selected_corners[0], (0, 255, 255), 2)
                print("\n4 corners captured! Press ANY KEY to calculate homography.")

            cv2.imshow("Select Pitch Corners", ui_image)


turf_img = cv2.imread("../inputs/turf.jpg")
flag_img = cv2.imread("../inputs/sri_lanka_flag.png")

if turf_img is None or flag_img is None:
    print("Error: Could not load images.")
    exit()

ui_image = turf_img.copy()
cv2.namedWindow("Select Pitch Corners")
cv2.setMouseCallback("Select Pitch Corners", capture_clicks)

print("Click 1: Top-Left | Click 2: Top-Right | Click 3: Bottom-Right | Click 4: Bottom-Left")
cv2.imshow("Select Pitch Corners", ui_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(selected_corners) != 4:
    print("Error: 4 points not selected.")
    exit()

dst_pts = np.array(selected_corners, dtype=np.float32)
fh, fw = flag_img.shape[:2]
src_pts = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1],
                   [0, fh - 1]], dtype=np.float32)

H_matrix, _ = cv2.findHomography(src_pts, dst_pts)
th, tw = turf_img.shape[:2]
warped_flag = cv2.warpPerspective(flag_img, H_matrix, (tw, th))

solid_white = np.ones((fh, fw), dtype=np.uint8) * 255

warped_mask = cv2.warpPerspective(solid_white, H_matrix, (tw, th))

mask_3c = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
inverse_mask_3c = cv2.bitwise_not(mask_3c)

background = cv2.bitwise_and(turf_img, inverse_mask_3c)

grass_under_flag = cv2.bitwise_and(turf_img, mask_3c)

painted_flag = cv2.addWeighted(warped_flag, 0.6, grass_under_flag, 0.4, 0)

final_output = cv2.add(background, painted_flag)

cv2.imshow("Final Superimposed Turf", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("q3_final_turf_srilanka.jpg", final_output)
print("Saved as 'q3_final_turf_srilanka.jpg'")
