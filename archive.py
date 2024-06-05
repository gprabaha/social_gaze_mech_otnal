#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:35:39 2024

@author: pg496
"""

"""
Archive of unused functions
"""

def is_inside_quadrilateral(point, corners, tolerance=1e-3):
    """
    Checks if a point is inside a quadrilateral.
    Parameters:
    - point (tuple): Point coordinates.
    - corners (dict): Dictionary containing corner coordinates.
    - tolerance (float): Tolerance level for area difference.
    Returns:
    - inside_quad (bool): True if the point is inside the quadrilateral, False otherwise.
    - area_diff (float): Difference in area.
    """
    x, y = point
    x1, y1 = corners['topLeft']
    x2, y2 = corners['topRight']
    x3, y3 = corners['bottomRight']
    x4, y4 = corners['bottomLeft']
    total_area = get_area_using_shoelace_4pts(x1, y1, x2, y2, x3, y3, x4, y4)
    triangle_area_point1 = get_area_using_shoelace_3pts(x, y, x1, y1, x2, y2)
    triangle_area_point2 = get_area_using_shoelace_3pts(x, y, x2, y2, x3, y3)
    triangle_area_point3 = get_area_using_shoelace_3pts(x, y, x3, y3, x4, y4)
    triangle_area_point4 = get_area_using_shoelace_3pts(x, y, x4, y4, x1, y1)
    sum_of_triangles = triangle_area_point1 + triangle_area_point2 + \
        triangle_area_point3 + triangle_area_point4
    area_diff = abs(total_area - sum_of_triangles)
    inside_quad = area_diff < tolerance
    return inside_quad, area_diff


def get_area_using_shoelace_3pts(x1, y1, x2, y2, x3, y3):
    """
    Calculates the area of a triangle using the Shoelace formula.
    Parameters:
    - x1, y1, x2, y2, x3, y3: Coordinates of the triangle vertices.
    Returns:
    - area: The area of the triangle.
    """
    return 0.5 * abs((x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1))


def get_area_using_shoelace_4pts(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Calculates the area of a quadrilateral using the Shoelace formula.
    Parameters:
    - x1, y1, x2, y2, x3, y3, x4, y4: Coordinates of the quadrilateral vertices.
    Returns:
    - area: The area of the quadrilateral.
    """
    total_area = get_area_using_shoelace_3pts(x1, y1, x2, y2, x3, y3) + \
                 get_area_using_shoelace_3pts(x1, y1, x3, y3, x4, y4)
    return total_area