import cv2
import numpy as np
import os
import fnmatch
import logging

"""
Finds corners in 121/293 images using findChessBoardCorners
Finds corners in 251/293 images using findChessBoardCorners

"""
def find_squares(img_path: str, visualize) -> None:
    # This method seems promising!!!
    # Followed code found here: https://medium.com/@siromermer/extracting-chess-square-coordinates-dynamically-with-opencv-image-processing-methods-76b933f0f64e
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gaussian_blur = cv2.GaussianBlur(gray_image, (5,5), 0)
    ret, otsu_binary = cv2.threshold(gaussian_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    canny = cv2.Canny(otsu_binary, 20, 255)
    kernel = np.ones((7,8), np.uint8)
    img_dilation = cv2.dilate(canny, kernel, iterations=1)
    lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, threshold=200, minLineLength=100, maxLineGap=50)

    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            cv2.line(img_dilation, (x1,y1), (x2,y2), (255,255,255), 2)

    kernel2 = np.ones((3,3), np.uint8)
    img_dilation2 = cv2.dilate(img_dilation, kernel2, iterations=1)

    board_contours, hierarchy = cv2.findContours(img_dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    square_centers = []
    board_squared = canny.copy()

    for contour in board_contours:
        # if 6000 < cv2.contourArea(contour) < 20000:
            eps = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)

            # Ensure approximated contour has 4 points
            if len(approx) == 4:
                pts = [pt[0] for pt in approx]
                pt1 = tuple(pts[0])
                pt2 = tuple(pts[1])
                pt3 = tuple(pts[2])
                pt4 = tuple(pts[3])

                x, y, w, h = cv2.boundingRect(contour)
                center_x = (x+(x+w))/2
                center_y = (y+(y+h))/2

                square_centers.append([center_x, center_y, pt2, pt1, pt3, pt4])

                # Draw lines between points
                cv2.line(board_squared, pt1, pt2, (255,255,0), 7)
                # cv2.line(board_squared, pt1, pt3, (255,255,0), 7)
                cv2.line(board_squared, pt2, pt4, (255,255,0), 7)
                cv2.line(board_squared, pt3, pt4, (255,255,0), 7)
                


    cv2.imshow("IM", board_squared)
    cv2.imshow("Original", image)
    cv2.waitKey(0)


def find_lines(img_path: str, visualize: bool = False) -> np.ndarray:
    """Find lines in given image, code from https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    Doens't work great."""
    im = cv2.imread(img_path)
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kernel_size = 13
    gray_blur = cv2.GaussianBlur(gray_im, (kernel_size, kernel_size), 0)
    low_lim, high_lim = 50, 150
    edges = cv2.Canny(gray_blur, low_lim, high_lim)

    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 200
    max_line_gap = 20
    line_img = np.copy(im) * 0

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if visualize:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1,y1), (x2,y2), (0, 0, 255), 5)
        lines_edges = cv2.addWeighted(im, 0.8, line_img, 1, 0)
        cv2.imshow("Image w/ lines", lines_edges)
        cv2.waitKey(0)

    return lines





def find_corners(img_path: str, visualize: bool = False) -> np.ndarray | None:
    """Given path to chessboard image, return coordinates of corners found in image.
    Return None if no corners found.
    
    @param img_path  Path to image
    @param visualize  Display image with corners if True"""
    im = cv2.imread(img_path)
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, corners = cv2.findChessboardCorners(gray_im, (7,7), None)
    ret, corners = cv2.findChessboardCornersSB(gray_im, (7,7), None)
    cv2.findChessboardCornersSB
    cv2.findChessboardCornersSBWithMeta

    if visualize:
        if corners is not None:
            for point in corners:
                corner_pixel = (int(point[0][0]), int(point[0][1]))
                im = cv2.circle(im, corner_pixel, radius=5, color=(0,0,255), thickness=-1)
            cv2.imshow("Image", im)
            cv2.waitKey(0)
        else:
            logging.warning("Tried to visualize image with corners, but none were found.")

    return corners

def square_board(img_path: str, visualize : bool = False) -> np.ndarray:
    """Given a path to a chessboard image, return openCV image warped such that the
    board is square.
      
    @param img_path  Path to image
    @param visualize  Display warped image if True"""
    pass


# Collect all available chessboard images
image_dir = '/Users/anton/Documents/Code/chess_vision/chess_vision/data/'
img_paths = []
for root, dirnames, filenames in  os.walk(image_dir):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        img_paths.append(os.path.join(root, filename))
    

if __name__ == '__main__':
    for i in range(10):
        find_squares(img_paths[i], visualize=True)
    # for img_path in img_paths:
    #     find_squares(img_path, visualize=True)

    # # Get number of images were edges are detected
    # num_imgs_with_corners = 0
    # for i, img_path in enumerate(img_paths):
    #     print(f"Img {i+1}/{len(img_paths)}", end='\r')
    #     if find_corners(img_path) is not None:
    #         # find_corners(img_path, visualize=True)
    #         num_imgs_with_corners += 1
    #     else:
    #         im = cv2.imread(img_path)
    #         cv2.imshow("Image", im)
    #         cv2.waitKey(0)

    # print(f"Corners were found in {num_imgs_with_corners}/{len(img_paths)} images")


# # Load the image, resize from (w,h) -> (w,h+offset) and make new pixels black
# # This is to make sure that no part of the image is cropped in upper corner
# # TODO: For some reason, only works on first image. For some images warping is wrong, for other it fails to find any corners.
# # Issue seems to be that the corner points can't be found for all images, issue seem to worsen
# # when there are more pieces on the board. I.e. for a fully set up board, no corners are found
# image_path = img_paths[20]
# image = cv2.imread(image_path)
# height, width = image.shape[:2]
# offset = 100
# new_height = height + offset
# resized_image = np.zeros((new_height, width, 3), dtype=np.uint8)
# resized_image[offset:, :] = image



# # Convert to grayscale
# gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# # cv2.waitKey(0)

# # Detect corners of the chessboard
# ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

# if corners is not None:
#     for point in corners:
#         print((point[0][0], point[0][1]))
#         resized_image = cv2.circle(resized_image, (int(point[0][0]), int(point[0][1])), radius=5, color=(0,0,255), thickness=-1)
#     cv2.imshow("Image",resized_image)
#     cv2.waitKey(0)
# else:
#     print("Could't display image, no corners found")


# # if ret:
# #     # Refine the corner locations
# #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# #     corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# #     # Define the four extreme corners of the chessboard
# #     top_right = corners[0][0]    
# #     bottom_right = corners[6][0] 
# #     bottom_left = corners[-1][0] 
# #     top_left = corners[-7][0]    

# #     # Define points for the perspective transformation
# #     pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

# #     # Define the desired output points to make the chessboard square
# #     # Keeping the board square but within the original image size
# #     height, width = resized_image.shape[:2]
# #     width = int(np.linalg.norm(top_right - top_left))
# #     height = int(np.linalg.norm(bottom_left - top_left))
# #     pts2 = np.float32([
# #         [top_left[0], top_left[1]],               # Top-left stays the same
# #         [top_left[0] + width, top_left[1]],       # Top-right adjusted horizontally
# #         [top_left[0] + width, top_left[1] + height],  # Bottom-right adjusted horizontally and vertically
# #         [top_left[0], top_left[1] + height]       # Bottom-left adjusted vertically
# #     ])

# #     # Compute the perspective transformation matrix
# #     matrix = cv2.getPerspectiveTransform(pts1, pts2)

# #     # Perform the perspective transformation, keeping the original image size
# #     transformed = cv2.warpPerspective(resized_image, matrix, (image.shape[1], image.shape[0]))

# #     # Save or display the result
# #     cv2.imshow('Transformed Chessboard', transformed)
# #     cv2.imshow('Original Chessboard', image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# # else:
# #     print("Chessboard corners not detected. Please try a clearer image.")
