# Main File of the gnomonic projection clustering algorithm
import itertools
import networkx as nx

from ildars.reflected_signal import ReflectedSignal
from ildars.clustering.cluster import ReflectionCluster
import ildars_visualization.gnomonic_projection as viz

from .arc import Arc
from .hemisphere import Hemisphere


def compute_reflection_clusters(reflected_signals, hemi_width_degree):
    hemispheres = Hemisphere.get_6_hemispheres()
    compute_stereographic_projection(reflected_signals, hemispheres, hemi_width_degree)
    return find_clusters(hemispheres)


def compute_stereographic_projection(reflected_signals, hemispheres, hemi_width_degree):
    arcs = [Arc(ref) for ref in reflected_signals]
    for arc in arcs:
        for hemi in hemispheres:
            hemi.add_arc(arc, hemi_width_degree)
    #for hemi in hemispheres:
    #    viz.plot_line_segments(hemi)


# Find the connected components on each hemisphere
def find_clusters(hemispheres):
    intersection_graphs = [
        hemi.get_intersection_graph() for hemi in hemispheres
    ]
    # Merge all graphs pair-wise
    while len(intersection_graphs) > 1:
        intersection_graphs = [
            nx.compose(*pair)
            for pair in itertools.pairwise(intersection_graphs)
        ]
    g = intersection_graphs[0]
    # Visualization
    viz.plot_hemisphere_connectivity_graph(g, "graph")

    # Get connected components
    components = list(nx.connected_components(g))
    # Filter out single nodes
    components = [c for c in components if len(c) > 1]
    clusters = [
        ReflectionCluster(
            [ReflectedSignal.Signals[index] for index in component]
        )
        for component in components
    ]
    return clusters
