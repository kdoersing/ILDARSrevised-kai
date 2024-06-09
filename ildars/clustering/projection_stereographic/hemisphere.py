import itertools
import numpy as np
import scipy as sp
import networkx as nx
import math
from skspatial.objects import Plane, Line, Circle

# only for debugging
import ildars_visualization.gnomonic_projection as viz

import ildars.math_utils as math_util
from . import util

# Threshold for detecting, whether a given arc is on a given hemisphere
# Threshold is directly taken from Rico Gießlers code, assuming the flag for
# 12 hemispheres to always be true
HEMI_WIDTH_DEGREE = 37.4
K_THRESHOLD = np.cos(np.radians(HEMI_WIDTH_DEGREE))
PHI = (1 + 5 ** 0.5) / 2

# Strings used for dictionaries
STR_START = "start"
STR_END = "end"
STR_ARC = "arc"


class Hemisphere:
    def __init__(self, center):
        self.center = math_util.normalize(center)
        self.lines = []

    # add arc to hemisphere, if its distance to the hemisphere center is below
    # the thrshold for cos_c value
    def add_arc(self, arc):
        start_lat_lon = util.carth_to_lat_lon(arc.start)
        end_lat_lon = util.carth_to_lat_lon(arc.end)
        hemi_lat_lon = util.carth_to_lat_lon(self.center)

        start_k = util.get_k(start_lat_lon, hemi_lat_lon)
        end_k = util.get_k(end_lat_lon, hemi_lat_lon)

        if start_k <= K_THRESHOLD and end_k <= K_THRESHOLD:
            # arc is not present on the hemisphere
            return

        if start_k <= K_THRESHOLD:
            new_start = self.clip_vector_to_hemisphere(arc.start, arc.end)
            start_lat_lon = util.carth_to_lat_lon(new_start)
            start_k = util.get_k(start_lat_lon, hemi_lat_lon)
        elif end_k <= K_THRESHOLD:
            new_end = self.clip_vector_to_hemisphere(arc.start, arc.end)
            end_lat_lon = util.carth_to_lat_lon(new_end)
            end_k = util.get_k(end_lat_lon, hemi_lat_lon)

        self.lines.append(
            {
                STR_START: util.lat_lon_to_stereographic(
                    start_lat_lon, hemi_lat_lon, start_k
                ),
                STR_END: util.lat_lon_to_stereographic(
                    end_lat_lon, hemi_lat_lon, end_k
                ),
                STR_ARC: arc,
            }
            # {
            #     "start": util.lat_lon_to_gnomonic(
            #         start_lat_lon, hemi_lat_lon, start_k
            #     ),
            #     "end": util.lat_lon_to_gnomonic(
            #         end_lat_lon, hemi_lat_lon, end_k
            #     ),
            #     # We also need to store a reference to the respective
            #     # reflected signal
            #     "reflection": arc.reflected_signal,
            # }
        )

    # given two vectors v_out, v_in where v_out is outside the hemisphere and
    # v_in is inside, adjust v_in s.t. it also lands on the hemisphere without
    # the line between v_in and v_out losing its direction
    def clip_vector_to_hemisphere(self, v_out, v_in):
        circle_center = np.array([0, 0, 0])
        circle_normal = math_util.normalize(np.cross(v_in, v_out))
        hemi_plane_center = K_THRESHOLD * self.center
        hemi_plane_normal = self.center
        cirle_plane = Plane(point=circle_center, normal=circle_normal)
        hemi_plane = Plane(point=hemi_plane_center, normal=hemi_plane_normal)
        cirle_hemi_intersection = cirle_plane.intersect_plane(hemi_plane)
        # get a new base of R^3 that is aligned to the circle plane
        cp_bv1 = math_util.normalize(v_in)
        cp_bv2 = math_util.normalize(np.cross(circle_normal, cp_bv1))
        cp_bv3 = circle_normal
        # now we get a base transformation matrix from our circle space to the
        # space with standard base
        mat_circ_std = np.transpose(np.array([cp_bv1, cp_bv2, cp_bv3]))
        assert np.linalg.matrix_rank(mat_circ_std) == 3
        mat_std_circ = np.linalg.inv(mat_circ_std)
        # now get two points of the intersection line and get coordinates
        # in our new "circle space"
        il_p1 = mat_std_circ.dot(cirle_hemi_intersection.point.to_array())
        il_p2 = mat_std_circ.dot(
            cirle_hemi_intersection.point.to_array()
            + cirle_hemi_intersection.direction.to_array()
        )
        # Now we get a new 2d space, where we simple set the third coordinate
        # of our "circle space" to 0
        il_p1 = il_p1[:-1]
        il_p2 = il_p2[:-1]
        # by construction, our circle goes through origin and has radius 1.
        # now we can intersect it with our new 2d line
        circ_2d = Circle([0, 0], 1)  # circle with radius 0 and center at (0,0)
        line_2d = Line(point=il_p1, direction=il_p2 - il_p1)
        intersections = circ_2d.intersect_line(line_2d)
        assert len(intersections) == 2
        # we now have two points on our 2d circle and select the right one.
        # To do so, we transform our original start and endpoint of our arc
        # and compare the polar coordinates
        arc_start_2d = mat_std_circ.dot(math_util.normalize(v_out))[:-1]
        # arc_end_2d = np.array([1, 0])  # we chose v_in as x-axis for our space
        # convert all points to polar coordinates, but ignoring the radius,
        # which is 1 for all points
        arc_start_phi = np.arctan2(arc_start_2d[1], arc_start_2d[0])
        # arc_end_phi = np.arctan2(arc_end_2d[1], arc_end_2d[0])
        p1_phi = np.arctan2(intersections[0][1], intersections[0][0])
        p2_phi = np.arctan2(intersections[1][1], intersections[1][0])
        # We need to differenciate between the two halves of the circle to
        # select the right intersection point
        solution_index = None
        if arc_start_phi < np.pi:
            if 0 <= p1_phi <= arc_start_phi:
                assert p2_phi > arc_start_phi
                solution_index = 0
            else:
                assert p2_phi <= arc_start_phi
                solution_index = 1
        else:
            if arc_start_phi <= p1_phi <= 2 * np.pi:
                assert p2_phi < arc_start_phi
                solution_index = 0
            else:
                assert p2_phi >= arc_start_2d
                solution_index = 1
        # Now transform solution back into our standard space
        solution = intersections[solution_index].to_array()
        solution = mat_circ_std.dot(np.array([solution[0], solution[1], 0]))

        return solution

    def get_intersection_graph(self):
        line_indeces = [
            line[STR_ARC].reflected_signal.index for line in self.lines
        ]
        g = nx.Graph()
        g.add_nodes_from(line_indeces)
        pairs = itertools.combinations(self.lines, 2)
        for pair in pairs:
            if util.intersect_2d(
                pair[0][STR_START],
                pair[0][STR_END],
                pair[1][STR_START],
                pair[1][STR_END],
            ):
                g.add_edge(
                    pair[0][STR_ARC].reflected_signal.index,
                    pair[1][STR_ARC].reflected_signal.index,
                )
        return g

    # Get 12 hemispheres with random initial orientation
    @staticmethod
    def get_12_hemispheres():
    #     dodecahedron_vertices = [
    #     [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    #     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    #     [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
    #     [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
    #     [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]]
    
        dodecahedron_faces = [
            [(1,1,1), (1,-1,1), (0,1/PHI,PHI), (0,-1/PHI,PHI), (PHI,0,1/PHI)],#oben rechts
            [(-1,1,1), (-1,-1,1), (0,1/PHI,PHI), (0,-1/PHI,PHI), (-PHI,0,1/PHI)],#oben links
            [(1,-1,1), (-1,-1,1), (0,-1/PHI,PHI), (1/PHI,-PHI,0), (-1/PHI,-PHI,0)],#nah oben
            [(1,1,1), (-1,1,1), (0,1/PHI,PHI), (1/PHI,PHI,0), (-1/PHI,PHI,0)],#weit oben
            [(-1,-1,1), (-1,-1,-1), (-1/PHI,-PHI,0), (-PHI,0, 1/PHI), (-PHI,0,-1/PHI)],#mitte links vorne
            [(1,-1,1), (1,-1,-1), (1/PHI,-PHI,0), (PHI,0,1/PHI), (PHI,0,-1/PHI)],#mitte rechts vorne
            [(-1,1,1), (-1,1,-1), (-1/PHI,PHI, 0), (-PHI,0,1/PHI), (-PHI,0,-1/PHI)],#mitte links hinten
            [(1,1,1), (1,1,-1), (1/PHI,PHI,0), (PHI,0,1/PHI), (PHI,0,-1/PHI)],#mitte hinten rechts
            [(1,-1,-1), (-1,-1,-1), (0,-1/PHI,-PHI), (1/PHI,-PHI,0), (-1/PHI,-PHI,0)],#nah unten
            [(1,1,-1), (1,-1,-1), (0,1/PHI,-PHI), (0,-1/PHI,-PHI), (PHI,0,-1/PHI)],#unten rechts
            [(-1,1,-1), (-1,-1,-1), (0,1/PHI,-PHI), (0,-1/PHI,-PHI), (-PHI,0,-1/PHI)],#unten links
            [(1,1,-1), (-1,1,-1), (0,1/PHI,-PHI), (1/PHI,PHI,0), (-1/PHI,PHI,0)]#weit unten
        ]
        # Calculate center points of the faces
        face_centers = []
        for face_vertices in dodecahedron_faces:
            # Convert vertices to NumPy array for easier calculations
            vertices_array = np.array(face_vertices)
            # Calculate the mean along each axis to get the center point
            center_point = np.mean(vertices_array, axis=0)
            face_centers.append(center_point)
        # Normalize the vertices to be on the unit sphere    
        normalized_vectors = [np.array(v) / np.linalg.norm(v) for v in face_centers]
        # randomly rotate all vectors
        rotation = sp.spatial.transform.Rotation.random()
        rotated_vectors = rotation.apply(normalized_vectors)
        return [Hemisphere(vec) for vec in rotated_vectors]
    

    def get_6_hemispheres():
        #cube_vertices = [
        # [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], 
        # [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]

        cube_faces = [
            [(1,1,1), (1,-1,1), (-1,1,1), (-1,-1,1)],#top
            [(1,1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,-1)],#bottom
            [(1,1,-1), (1,-1,-1), (1,1,1), (1,-1,1)],#right
            [(1,1,-1), (-1,1,-1), (1,1,1), (-1,1,1)],#back
            [(-1,1,-1), (-1,-1,-1), (-1,1,1), (-1,-1,1)],#left
            [(1,-1,-1), (-1,-1,-1), (1,-1,1), (-1,-1,1)]#front
        ]
        # Calculate center points of the faces
        face_centers = []
        for face_vertices in cube_faces:
            # Convert vertices to NumPy array for easier calculations
            vertices_array = np.array(face_vertices)
            # Calculate the mean along each axis to get the center point
            center_point = np.mean(vertices_array, axis=0)
            face_centers.append(center_point)
        # Normalize the vertices to be on the unit sphere    
        normalized_vectors = [np.array(v) / np.linalg.norm(v) for v in face_centers]
        # randomly rotate all vectors
        rotation = sp.spatial.transform.Rotation.random()
        rotated_vectors = rotation.apply(normalized_vectors)
        return [Hemisphere(vec) for vec in rotated_vectors]


    def get_4_hemispheres():
        #tetrahedron_vertices = [
        # [1,0,-1/math.sqrt(2)], [-1,0,-1/math.sqrt(2), 
        # [0,1,1/math.sqrt(2)], [0,-1,1/math.sqrt(2)]]

        tetrahedron_faces = [
            [(1,0,-1/math.sqrt(2)), (-1,0,-1/math.sqrt(2)), (0,1,1/math.sqrt(2))],
            [(1,0,-1/math.sqrt(2)), (-1,0,-1/math.sqrt(2)), (0,-1,1/math.sqrt(2))],
            [(1,0,-1/math.sqrt(2)), (0,1,1/math.sqrt(2)), (0,-1,1/math.sqrt(2))],
            [(-1,0,-1/math.sqrt(2)), (0,1,1/math.sqrt(2)), (0,1,1/math.sqrt(2))],
        ]
        # Calculate center points of the faces
        face_centers = []
        for face_vertices in tetrahedron_faces:
            # Convert vertices to NumPy array for easier calculations
            vertices_array = np.array(face_vertices)
            # Calculate the mean along each axis to get the center point
            center_point = np.mean(vertices_array, axis=0)
            face_centers.append(center_point)
        # Normalize the vertices to be on the unit sphere    
        normalized_vectors = [np.array(v) / np.linalg.norm(v) for v in face_centers]
        # randomly rotate all vectors
        rotation = sp.spatial.transform.Rotation.random()
        rotated_vectors = rotation.apply(normalized_vectors)
        return [Hemisphere(vec) for vec in rotated_vectors]


    def get_3_hemispheres():
        vectors = [np.array([-1,0,0]), np.array([0.5,-math.sqrt(3)/2,0]), np.array([0.5,math.sqrt(3)/2,0])]

        # Normalize the vertices to be on the unit sphere    
        normalized_vectors = [np.array(v) / np.linalg.norm(v) for v in vectors]
        # randomly rotate all vectors
        rotation = sp.spatial.transform.Rotation.random()
        rotated_vectors = rotation.apply(normalized_vectors)
        return [Hemisphere(vec) for vec in rotated_vectors]
    

    def get_2_hemispheres():
        vectors = [np.array([0,0,1]), np.array([0,0,-1])]

        # Normalize the vertices to be on the unit sphere    
        normalized_vectors = [np.array(v) / np.linalg.norm(v) for v in vectors]
        # randomly rotate all vectors
        rotation = sp.spatial.transform.Rotation.random()
        rotated_vectors = rotation.apply(normalized_vectors)
        return [Hemisphere(vec) for vec in rotated_vectors]
