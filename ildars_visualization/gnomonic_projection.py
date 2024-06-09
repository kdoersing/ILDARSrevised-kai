import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skspatial.objects import Point, LineSegment


def plot_clipping(
    circ_2d, line_2d, intersections, solution_index, arc_start_2d, arc_end_2d
):
    _, ax = plt.subplots()
    circ_2d.plot_2d(ax, fill=False)
    line_2d.plot_2d(ax, t_2=5, c="k")
    intersections[solution_index].plot_2d(ax, s=100, c="g")
    intersections[np.abs(solution_index - 1)].plot_2d(ax, s=100, c="r")
    Point(arc_start_2d).plot_2d(ax, s=100, c="b")
    Point(arc_end_2d).plot_2d(ax, s=100, c="b")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.show()


def plot_hemisphere_connectivity_graph(g, name):
    nx.draw(g, with_labels=True)
    plt.savefig(f"{name}.svg")

def plot_line_segments(hemisphere):
        plt.figure()
        #print(hemisphere.lines)
        for line in hemisphere.lines:
            plt.plot([line["start"][0], line["start"][1]], [line["end"][0], line["end"][1]], color="green", marker='o')

             

        # Add labels and title
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('2D Plot of Line Segments')
        #plt.legend()

        # Save the plot as an SVG file
        plt.savefig('line_segments_plot.svg')

        # Show the plot
        plt.show()