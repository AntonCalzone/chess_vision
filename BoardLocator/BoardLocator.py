import cv2
import os
import fnmatch
import numpy as np

from .utils import pairwise_dist, linear_regression, contour_square


class BoardLocator:

    def __init__(self) -> None:
        """Constructor for BoarLocator class."""
        # Intermediary images
        self.images = {
            'hough_lines': None,
            'squares': None,
            'numbered_squares': None,
            'row_lines': None
        }

        # Parameters
        # TODO: These should be specified in a parameters file instead, and accessed via 
        # a parameters class instance.
        self.hough_line_min_length = 700
        self.min_contour_area = 6000
        self.max_contour_area = 55000
        self.min_square_ratio = 0.5
        self.linear_reg_threshold = 0.97

    def find_squares(self, image: np.ndarray) -> list:
        """Given an image of a chess board, return the positions of the squares
        as a list. Heavily inspired by the following article:
        https://medium.com/@siromermer/extracting-chess-square-coordinates-dynamically-with-opencv-image-processing-methods-76b933f0f64e
        
        Params:
            image: Input image"""
        gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add Gaussian vlur to reduce noise and smooth image
        gaussian_blur = cv2.GaussianBlur(gray_im, (5,5), 0)

        # OTSU threshold to get binary image
        ret, otsu_binary = cv2.threshold(gaussian_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Canny edge detection
        canny = cv2.Canny(otsu_binary, 20, 255)

        # Dilation
        kernel = np.ones((7,7), np.uint8)
        img_dilation = cv2.dilate(canny, kernel, iterations=1)

        # Hough lines (straighten lines)
        lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, threshold=200,
                                minLineLength=100, maxLineGap=50)
        
        # Remove lines that are too short, threshold found empirically
        hough_lines_im = np.zeros_like(img_dilation)
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                length = pairwise_dist((x1, y1), (x2, y2))
                if length > self.hough_line_min_length:
                    cv2.line(hough_lines_im, (x1, y1), (x2, y2), (255, 255, 255), 2)
        self.images['hough_lines'] = hough_lines_im
        self.hough_lines_im = hough_lines_im

        # Second dilation, unsure of why
        kernel = np.ones((3, 3), np.uint8)
        img_dilation_2 = cv2.dilate(hough_lines_im, kernel, iterations=1)

        # Find and filter contours
        board_contours, hierarcy = cv2.findContours(img_dilation_2, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        square_centers = []
        squares_im = np.zeros_like(hough_lines_im)

        for i, contour in enumerate(board_contours):
            if self.min_contour_area < cv2.contourArea(contour) < self.max_contour_area:
                # Sinmplify contour
                eps = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, eps, True)
                if len(approx) != 4:
                    continue

                pts = [pt[0] for pt in approx]
                pt1 = tuple(pts[0])
                pt2 = tuple(pts[1])
                pt4 = tuple(pts[2])
                pt3 = tuple(pts[3])

                if not contour_square(pt1, pt2, pt3, pt4, self.min_square_ratio):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center_x = (x + (x + w)) / 2
                center_y = (y + (y + h)) / 2
                square_centers.append([center_x, center_y, pt2, pt1, pt3, pt4])

                cv2.line(squares_im, pt1, pt2, (255, 255, 0), 7)
                cv2.line(squares_im, pt1, pt3, (255, 255, 0), 7)
                cv2.line(squares_im, pt2, pt4, (255, 255, 0), 7)
                cv2.line(squares_im, pt3, pt4, (255, 255, 0), 7)

        self.images['squares'] = squares_im

        # Sort squares by center values
        sorted_coords = sorted(square_centers, key=lambda x: x[1], reverse=True)
        groups = []
        current_group = [sorted_coords[0]]

        for coord in sorted_coords[1:]:
            if abs(coord[1] - current_group[-1][1]) < 50:
                current_group.append(coord)
            else:
                groups.append(current_group)
                current_group = [coord]
        
        groups.append(current_group)
        for group in groups:
            group.sort(key=lambda x: x[0])

        sorted_coords = [coord for group in groups for coord in group]

        # Number squares
        numbered_squares = squares_im.copy()
        square_num = 1
        for coord in sorted_coords:
            cv2.putText(img=numbered_squares, text=str(square_num),
                        org=(int(coord[0])-30, int(coord[1])), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1, color=(125, 246, 55), thickness=3)
            square_num += 1
        self.images['numbered_squares'] = numbered_squares



        # Map square coords to horizontal lines for each row
        row_lines = []
        row_lines_im = squares_im.copy()
        xs = np.array([square[0] for square in sorted_coords])
        i_row_breaks = np.where(xs - np.array([*xs[1:], xs[-1]]) >= 0)[0] + 1
        i_start = 0
        for i_end in i_row_breaks:
            _, intercept, coef = linear_regression(sorted_coords, i_start, i_end)
            row_lines.append((intercept, coef))
            p1 = (1, int(coef + intercept))
            p2 = (1919, int(coef * 1919 + intercept))
            cv2.line(row_lines_im, p1, p2, (125, 246, 55), 3)
            i_start = i_end

        coefs = [round(coef, 3) for _, coef in row_lines]
        print(coefs)





        # Map square coord to horizontal lines
        # row_lines_im = squares_im.copy()
        # row_lines = []
        # i_start = 0
        # i_end = 3
        # while i_end < len(sorted_coords):
        #     r_sq, intercept, coef = linear_regression(sorted_coords, i_start, i_end)

        #     # Line found last iteration
        #     if (r_sq < self.linear_reg_threshold) and ((i_end - i_start) > 3) and (abs(coef[0]) < 0.1):
        #         _, intercept, coef = linear_regression(sorted_coords, i_start, i_end - 1)
        #         p1 = (1, int(coef[0] + intercept))
        #         p2 = (1919, int(coef[0] * 1919 + intercept))
        #         cv2.line(row_lines_im, p2, p1, (255, 255, 255), 3)
        #         row_lines.append((intercept, coef[0]))

        #         # for i in range(i_start, i_end):
        #         #     x, y, _, _, _, _ = sorted_coords[i]
        #         #     perp_coef = -1 / coef[0]
        #         #     # perp_intercept = x * (coef[0] + 1 / coef[0]) + intercept
        #         #     perp_intercept = y + 1 / coef[0] * x
        #         #     p1 = (1, int(perp_coef + intercept))
        #         #     p2 = (1079, int(perp_coef * 1079 + perp_intercept))
        #         #     cv2.line(row_lines_im, p1, p2, (255, 255, 255), 3)

        #         i_start = i_end + 1
        #         i_end = i_start + 3
        #     # No line could be constructed
        #     elif r_sq < self.linear_reg_threshold:
        #         i_start += 1
        #         i_end += 1
        #     # Try construct line with one more point
        #     else:
        #         # Except if last point, then try to construct line now
        #         if i_end + 1 == len(sorted_coords) and (i_end - i_start) > 3:
        #             _, intercept, coef = linear_regression(sorted_coords, i_start, i_end)
        #             p1 = (1, int(coef[0] + intercept))
        #             p2 = (1919, int(coef[0] * 1919 + intercept))
        #             cv2.line(row_lines_im, p2, p1, (255, 255, 255), 3)
        #             row_lines.append((intercept, coef[0]))
        #         i_end += 1

        self.images['row_lines'] = row_lines_im

        print(f"Found {len(row_lines)} lines")





        

