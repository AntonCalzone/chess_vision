import cv2
import numpy as np

# Load the image, resize from (w,h) -> (w,h+offset) and make new pixels black
# This is to make sure that no part of the image is cropped in upper corner
image_path = '/Users/anton/Documents/Code/chess_vision/chess_vision/data/Chess Pieces.v23-raw.yolov8/test/images/0b47311f426ff926578c9d738d683e76_jpg.rf.0b55f43ac16aa65c889558d8ea757072.jpg'
image = cv2.imread(image_path)
height, width = image.shape[:2]
offset = 100
new_height = height + offset
resized_image = np.zeros((new_height, width, 3), dtype=np.uint8)
resized_image[offset:, :] = image


# Convert to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Detect corners of the chessboard
ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

if ret:
    # Refine the corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Define the four extreme corners of the chessboard
    top_right = corners[0][0]    
    bottom_right = corners[6][0] 
    bottom_left = corners[-1][0] 
    top_left = corners[-7][0]    

    # Define points for the perspective transformation
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

    # Define the desired output points to make the chessboard square
    # Keeping the board square but within the original image size
    height, width = resized_image.shape[:2]
    width = int(np.linalg.norm(top_right - top_left))
    height = int(np.linalg.norm(bottom_left - top_left))
    pts2 = np.float32([
        [top_left[0], top_left[1]],               # Top-left stays the same
        [top_left[0] + width, top_left[1]],       # Top-right adjusted horizontally
        [top_left[0] + width, top_left[1] + height],  # Bottom-right adjusted horizontally and vertically
        [top_left[0], top_left[1] + height]       # Bottom-left adjusted vertically
    ])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Perform the perspective transformation, keeping the original image size
    transformed = cv2.warpPerspective(resized_image, matrix, (image.shape[1], image.shape[0]))

    # Save or display the result
    cv2.imshow('Transformed Chessboard', transformed)
    # cv2.imshow('Original Chessboard', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard corners not detected. Please try a clearer image.")
