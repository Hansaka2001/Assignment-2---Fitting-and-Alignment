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


turf_img = cv2.imread("turf.jpg")
flag_img = cv2.imread("sri_lanka_flag.png")

if turf_img is None or flag_img is None:
    print("Error: Make sure 'turf.jpg' and 'sri_lanka_flag.png' are in the same folder as this script.")
    exit()

ui_image = turf_img.copy()
cv2.namedWindow("Select Pitch Corners")
cv2.setMouseCallback("Select Pitch Corners", capture_clicks)

print("INSTRUCTIONS:")
print("Click the 4 corners of the cricket pitch in this EXACT order:")
print("1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")

cv2.imshow("Select Pitch Corners", ui_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(selected_corners) != 4:
    print("Error: You must select exactly 4 points. Please run the script again.")
    exit()


dst_pts = np.array(selected_corners, dtype=np.float32)

fh, fw = flag_img.shape[:2]
src_pts = np.array([
    [0, 0],
    [fw - 1, 0],
    [fw - 1, fh - 1],
    [0, fh - 1]
], dtype=np.float32)

H_matrix, _ = cv2.findHomography(src_pts, dst_pts)

th, tw = turf_img.shape[:2]
warped_flag = cv2.warpPerspective(flag_img, H_matrix, (tw, th))

flag_gray = cv2.cvtColor(warped_flag, cv2.COLOR_BGR2GRAY)
_, flag_mask = cv2.threshold(flag_gray, 1, 255, cv2.THRESH_BINARY)
inverse_mask = cv2.bitwise_not(flag_mask)

background = cv2.bitwise_and(turf_img, turf_img, mask=inverse_mask)

grass_under_flag = cv2.bitwise_and(turf_img, turf_img, mask=flag_mask)

paint_opacity = 0.6
painted_flag = cv2.addWeighted(
    warped_flag, paint_opacity, grass_under_flag, 1 - paint_opacity, 0)

final_output = cv2.add(background, painted_flag)

cv2.imshow("Final Superimposed Turf", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("q3_final_turf.jpg", final_output)
print("Saved image as 'q3_final_turf.jpg'.")
