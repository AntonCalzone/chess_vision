import numpy as np
from sklearn.linear_model import LinearRegression


def pairwise_dist(p1: tuple[float], p2: tuple[float]) -> float:
    """Given two 2D-points represented as tuples, return the distance
    between them."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def linear_regression(coords: list[list], i_start: int, i_end: int) -> tuple[float, float, float]:
    """Return the coefficient of determination, slope, and intercept of the line
    passing through all points from i_start to i_end in the given list of squares."""
    x = np.array([coords[i][0] for i in range(i_start, i_end)]).reshape((-1, 1))
    y = np.array([coords[i][1] for i in range(i_start, i_end)])
    model = LinearRegression().fit(x, y)
    return model.score(x, y), model.intercept_, model.coef_

def contour_square(pt1: tuple[float], pt2: tuple[float], pt3: tuple[float], pt4: tuple[float],
                   min_diff: float) -> bool:
        """Check if specified rectangle is approximately square according to supplied 
        max threshold."""
        side_lengths = [pairwise_dist(pt1, pt2), pairwise_dist(pt1, pt3),
                        pairwise_dist(pt2, pt4), pairwise_dist(pt3, pt4)]
        side_lengths.sort()
        short_sides = np.mean(side_lengths[:2])
        long_sides = np.mean(side_lengths[2:])
        diff = abs(short_sides / long_sides)
        return diff > min_diff