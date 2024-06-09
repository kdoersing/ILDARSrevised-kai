# Collection of helper functions for gnomonic projection
import numpy as np
from skspatial.objects import LineSegment


# get latitude and longituede in radians for given vector
def carth_to_lat_lon(v):
    lat = np.arcsin(v[2])
    lon = np.arctan2(v[0], v[1])
    return (lat, lon)


def get_k(v_ll, center_ll):
    v_lat = v_ll[0]
    v_lon = v_ll[1]
    center_lat = center_ll[0]
    center_lon = center_ll[1]
    return 2 / (
        1 + np.sin(center_lat) * np.sin(v_lat) + np.cos(center_lat) * np.cos(v_lat) * np.cos(v_lon - center_lon)
    )
    #return 2 / (1 + np.sin(center_lat))


def lat_lon_to_stereographic(ll_point, ll_hemi, k_point):
    lat_hemi = ll_hemi[0]
    lon_hemi = ll_hemi[1]
    lat_point = ll_point[0]
    lon_point = ll_point[1]
    x = (
        k_point 
        * np.cos(np.radians(lat_point)) 
        * np.sin(np.radians(lon_point - lon_hemi))
    )
    y = (
        k_point
        * (
            np.cos(np.radians(lat_hemi)) 
            * np.sin(np.radians(lat_point))
            - np.sin(np.radians(lat_hemi))
            * np.cos(np.radians(lat_point))
            * np.cos(np.radians(lon_point - lon_hemi))
        )
    )
    #x = k_point * (np.cos(lat_point - lat_hemi) * np.cos(lon_point - lon_hemi) / 1 - np.sin(lat_point - lat_hemi))
    #y = k_point * (np.cos(lat_point - lat_hemi) * np.sin(lon_point - lon_hemi) / 1 - np.sin(lat_point - lat_hemi))
    return (x, y)


# return true if two given lines intersect
def intersect_2d(l1p1, l1p2, l2p1, l2p2):
    l1 = LineSegment(l1p1, l1p2)
    l2 = LineSegment(l2p1, l2p2)
    try:
        _ = l1.intersect_line_segment(l2)
    except ValueError:
        return False
    return True
