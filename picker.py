import cv2
import sys
import os

def pick_coordinates(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not decode image.")
        return

    print("\n--- Interactive Coordinate Picker ---")
    print("Click anywhere on the image to get (Row, Col) for targets.")
    print("Press 'q' or 'ESC' to exit.\n")

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Note: OpenCV x,y maps to Col, Row
            print(f"Target Picked:  Row={y}, Col={x}")
            # Visual feedback
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Coordinate Picker", img)

    cv2.namedWindow("Coordinate Picker", cv2.WINDOW_NORMAL)
    # Resize window to a reasonable size but keep original coordinate mapping
    h, w = img.shape[:2]
    # Simple aspect ratio scaling for large images
    max_dim = 1000
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        cv2.resizeWindow("Coordinate Picker", int(w * scale), int(h * scale))
        
    cv2.setMouseCallback("Coordinate Picker", click_event)
    cv2.imshow("Coordinate Picker", img)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 picker.py <path_to_image>")
    else:
        pick_coordinates(sys.argv[1])
