import os
import fnmatch
import cv2

from BoardLocator.BoardLocator import BoardLocator



image_dir = '/Users/anton/Documents/Code/chess_vision/chess_vision/data/'
img_paths = []
for root, dirnames, filenames in  os.walk(image_dir):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        img_paths.append(os.path.join(root, filename))

board_locator = BoardLocator()

for im_path in img_paths:
    try:
        im = cv2.imread(im_path)
        board_locator.find_squares(im)

        # cv2.imshow("Hough lines", board_locator.images['hough_lines'])
        # cv2.imshow("Squares", board_locator.images['squares'])
        # cv2.imshow("Squares", board_locator.images['numbered_squares'])
        cv2.imshow("Squares", board_locator.images['row_lines'])
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    except IndexError as e:
        print(f"Error: {e}")
