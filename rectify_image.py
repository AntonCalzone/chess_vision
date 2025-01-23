# import cv2 as cv
# import numpy as np


# # Followed OpenCV tutorial for undistorting images
# # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# im_path = '/Users/anton/Documents/Code/chess_vision/chess_vision/data/Chess Pieces.v23-raw.yolov8/test/images/0b47311f426ff926578c9d738d683e76_jpg.rf.0b55f43ac16aa65c889558d8ea757072.jpg'

# # Find image and world points
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((6*7, 3), np.float32)
# objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# obj_points = []
# img_points = []

# img = cv.imread(im_path)
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, corners = cv.findChessboardCorners(gray_img, (7,6), None)
# if ret:
#     obj_points.append(objp)
#     corners2 = cv.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
#     img_points.append(corners2)

#     # cv.drawChessboardCorners(img, (7, 6), corners2, ret)
#     # cv.imshow('img', img)
#     # cv.waitKey(5000)

# # Get camera matrix
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)

# # Undistort image
# img = cv.imread(im_path)
# h, w = img.shape[:2]
# new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# undist_img = cv.undistort(img, mtx, dist, None, new_camera_matrix)
# x, y, w, h = roi 
# # undist_img = undist_img[y:y+h, x:x+w]
# cv.imshow('Undistorted', undist_img)
# cv.waitKey(5000)


import cv2
import numpy as np

# Load the image
image_path = '/Users/anton/Documents/Code/chess_vision/chess_vision/data/Chess Pieces.v23-raw.yolov8/test/images/0b47311f426ff926578c9d738d683e76_jpg.rf.0b55f43ac16aa65c889558d8ea757072.jpg'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    transformed = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    print(f"Size original image: {image.shape}")
    print(f"Size transformed image: {transformed.shape}")

    # Save or display the result
    cv2.imshow('Transformed Chessboard', transformed)
    cv2.imshow('Original Chessboard', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard corners not detected. Please try a clearer image.")
