import cv2

# --- CONFIGURATION ---
# Replace this with the path to one of your "problematic" dark images
IMAGE_PATH = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/pass2.jpg"


def get_pixel_value(event, x, y, flags, param):
    """
    Callback function that triggers on mouse clicks.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the intensity from the grayscale image
        # x is column (width), y is row (height)
        intensity = gray_img[y, x]

        print(f"Clicked at (x={x}, y={y}) | Grayscale Value: {intensity}")

        # Guide for the user
        if intensity == 0:
            print("  -> Pure Black")
        elif intensity < 10:
            print("  -> Very deep black (likely compression artifacts)")
        elif intensity < 50:
            print("  -> Dark Gray/Shadow")
        else:
            print("  -> Light area")


# 1. Load the image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"Error: Could not find or open image at {IMAGE_PATH}")
else:
    # 2. Convert to grayscale (since your filter script uses grayscale)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Create a window and set the mouse callback
    cv2.namedWindow("Pixel Checker")
    cv2.setMouseCallback("Pixel Checker", get_pixel_value)

    print("--- INSTRUCTIONS ---")
    print("1. Click anywhere on the image to see the darkness value.")
    print("2. Values closer to 0 are darker.")
    print("3. Press any key on your keyboard to close the window.")

    # 4. Show the image and wait for a keypress
    cv2.imshow("Pixel Checker", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
