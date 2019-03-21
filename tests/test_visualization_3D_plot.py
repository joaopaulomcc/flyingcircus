import numpy as np
import matplotlib.pyplot as plt

from pyquaternion import Quaternion

from context import flyingcircus
from flyingcircus import geometry as geo
from flyingcircus import visualization as vis


def test_plot_node():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    xyz_1 = np.array([0, 0, 0])
    quaternion_1 = Quaternion(axis=[1, 1, 1], angle=np.pi/3)

    xyz_2 = np.array([10, 0, 0])
    quaternion_2 = Quaternion(axis=[1, 0, 0], angle=np.pi/2)

    node_1 = geo.objects.Node(xyz_1, quaternion_1)
    # node_2 = geo.objects.Node(xyz_2, quaternion_2)

    rotation_quaternion = Quaternion(axis=[0, 0, 1], angle=np.pi/3)
    rotation_center = [2, 2, 0]

    node_2 = node_1.rotate(rotation_quaternion=rotation_quaternion, rotation_center=rotation_center)

    vis.plot_3D.plot_node(node_1, ax)
    vis.plot_3D.plot_node(node_2, ax)

    ax.scatter(rotation_center[0], rotation_center[1], rotation_center[2])

#    n_nodes = 20
#
#    nodes_list = geo.functions.interpolate_nodes(node_1, node_2, n_nodes)
#
#    for node_prop in nodes_list:
#
#        node = geo.objects.Node(node_prop[0], node_prop[1])
#        vis.plot_3D.plot_node(node, ax)
#
#    ax.set_xlabel("X axis")
#    ax.set_ylabel("Y axis")
#    ax.set_zlabel("Z axis")

    vis.plot_3D.set_axes_equal(ax)
    plt.show()


# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    test_plot_node()
