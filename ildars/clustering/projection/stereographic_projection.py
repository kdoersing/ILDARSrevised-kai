import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def getRandomPointOnUnitSphere():
    # Generate random spherical coordinates
    theta = 2 * np.pi * np.random.random()  # theta is a random angle between 0 and 2pi
    phi = np.pi / 2 * np.random.random()  # phi is a random angle between 0 and pi/2
    #phi = np.arccos(2 * np.random.random() - 1)  # phi is a random angle between 0 and pi
    #phi = np.pi / 2 + np.pi / 2 * np.random.random()  # phi is a random angle between pi/2 and pi

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    rand_point = np.array([x, y, z])
    if rand_point[0]**2 + rand_point[1]**2 + rand_point[2]**2 != 1 or (rand_point[0] == 0 and rand_point[1] == 0 and rand_point[2] == 1):
        return getRandomPointOnUnitSphere()

    return np.array([x, y, z])


def stereographic_projection(vector, z_height):
    # Project the point on the plane z = 0
    x_proj = vector[0] / (z_height + 1 - vector[2])
    y_proj = vector[1] / (z_height + 1 - vector[2])

    return np.array([x_proj, y_proj])


def main():
    N = np.array([0, 0, 1])
    z_height = -2

    random_points = []
    for i in range(1000):
        random_points.append(getRandomPointOnUnitSphere())
    random_points_x = []
    random_points_y = []
    random_points_z = []
    for point in random_points:
        point_projected = stereographic_projection(point, z_height)
        random_points_x.append(point_projected[0])
        random_points_y.append(point_projected[1])
        random_points_z.append(z_height)
    random_points_unprojected_x = [0]
    random_points_unprojected_y = [0]
    random_points_unprojected_z = [1]
    for point in random_points:
        random_points_unprojected_x.append(-point[0])
        random_points_unprojected_y.append(-point[1])
        random_points_unprojected_z.append(-point[2])

    rand_point = getRandomPointOnUnitSphere()
    print(rand_point[0]**2 + rand_point[1]**2 + rand_point[2]**2)
    print(stereographic_projection(rand_point, z_height))

    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    ticks_frequency = 1

    # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the plane z = -1
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    zz = z_height * np.ones(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    # Plot the points
    ax.scatter(random_points_unprojected_x, random_points_unprojected_y, random_points_unprojected_z, color='blue')
    ax.scatter(random_points_x, random_points_y, random_points_z, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    plt.show()


if __name__ == "__main__":
    main()
