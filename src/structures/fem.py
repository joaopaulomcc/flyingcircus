import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from pyquaternion import Quaternion

from . import functions as f
from . import objects as o
from .. import mathematics as m


def number_nodes(components_list, components_nodes_list, connections_list):
    """This function modifies the number attribute in the nodes of the componentes nodes lists
    """

    # Generate the connection matrix
    connection_matrix = []
    for connection in connections_list:
        connection_matrix.append(connection.descriptor)

    node_counter = 0

    for i, component in enumerate(components_list):

        for j, node in enumerate(components_nodes_list[i]):

            # If the node is the root of the component check if it is connected to a node tha has
            # already been numbered
            if j == 0:
                # Check if node is part of a connection
                node_identifier = component.identifier + "-ROOT"
                connections = check_connections(
                    connections_list, components_list, node_identifier
                )

                # Run through the connections and selects connected nodes
                connected_nodes = []
                for connection in connections:

                    connected_comp_index = components_list.index(connection[0])
                    connected_comp_nodes = components_nodes_list[connected_comp_index]

                    if connection[1] == "ROOT":
                        connected_node = connected_comp_nodes[0]

                    elif connection[1] == "TIP":
                        last_index = len(connected_comp_nodes) - 1
                        connected_node = connected_comp_nodes[last_index]

                    connected_nodes.append(connected_node)

                # Run through the connected nodes and checks if one of the is numbered
                node_number = None
                for connected_node in connected_nodes:
                    if connected_node.number is not None:
                        node_number = connected_node.number

                if node_number is not None:
                    node.number = node_number
                else:
                    node.number = node_counter
                    node_counter += 1

            # If the node is the tip of the component check if it is connected to a node tha has
            # already been numbered, same as the root
            if j == len(components_nodes_list[i]) - 1:
                # Check if node is part of a connection
                node_identifier = component.identifier + "-TIP"
                connections = check_connections(
                    connections_list, components_list, node_identifier
                )

                # Run through the connections and selects connected nodes
                connected_nodes = []
                for connection in connections:

                    connected_comp_index = components_list.index(connection[0])
                    connected_comp_nodes = components_nodes_list[connected_comp_index]

                    if connection[1] == "ROOT":
                        connected_node = connected_comp_nodes[0]

                    elif connection[1] == "TIP":
                        last_index = len(connected_comp_nodes) - 1
                        connected_node = connected_comp_nodes[last_index]

                    connected_nodes.append(connected_node)

                # Run through the connected nodes and checks if one of the is numbered
                node_number = None
                for connected_node in connected_nodes:
                    if connected_node.number is not None:
                        node_number = connected_node.number

                if node_number is not None:
                    node.number = node_number
                else:
                    node.number = node_counter
                    node_counter += 1

            # If the node is not at the ROOT or the TIP give it the next avaiable number:
            else:
                node.number = node_counter
                node_counter += 1


# ==================================================================================================


def check_connections(connections_list, components_list, node_identifier):
    """This function returns all the connections to a node
    """
    connections = []

    # Create a list with all the connection identifiers
    connection_matrix = []
    for connection in connections_list:
        connection_matrix.append(connection.descriptor)

    # Run through the connections
    for i, connection in enumerate(connection_matrix):

        # Finds a connection that contains the node identifier
        if node_identifier in connection:

            # Connections contains two components, index tells if it is the first or second
            # component
            index = connection.index(node_identifier)
            connected_component_index = abs(index - 1)
            connected_component = connections_list[i][connected_component_index * 2]
            connected_node = connections_list[i][connected_component_index * 2 + 1]

            connections.append([connected_component, connected_node])

    return connections


# ==================================================================================================


def create_macrosurface_connections(macrosurface):

    if macrosurface.symmetry_plane == "XZ" or macrosurface.symmetry_plane == "xz":

        middle_index = int(len(macrosurface.surface_list) / 2)

    else:
        middle_index = 0

    connections_list = []

    if middle_index == 0:

        for i in range(len(macrosurface.surface_list) - 1):

            connection = o.Connection(
                component1=macrosurface.surface_list[i],
                component1_node="TIP",
                component2=macrosurface.surface_list[i + 1],
                component2_node="ROOT",
            )

            connections_list.append(connection)

    else:
        for i in range(len(macrosurface.surface_list) - 1):

            if i >= middle_index:
                connection = o.Connection(
                    component1=macrosurface.surface_list[i],
                    component1_node="TIP",
                    component2=macrosurface.surface_list[i + 1],
                    component2_node="ROOT",
                )

            elif i == middle_index - 1:
                connection = o.Connection(
                    component1=macrosurface.surface_list[i],
                    component1_node="ROOT",
                    component2=macrosurface.surface_list[i + 1],
                    component2_node="ROOT",
                )

            else:
                connection = o.Connection(
                    component1=macrosurface.surface_list[i],
                    component1_node="ROOT",
                    component2=macrosurface.surface_list[i + 1],
                    component2_node="TIP",
                )

            connections_list.append(connection)


# ==================================================================================================


def generate_macrosurface_fem_elements(macrosurface, macrosurface_nodes_list, prop_choice="MIDDLE"):
    """Generates the beam finite elements of a macrosurface object.

    Args

    Returns
    """

    macrosurface_elements = []

    for surface, surface_nodes_list in zip(macrosurface, macrosurface_nodes_list):

        n_nodes = len(surface_nodes_list)

        # Calculates the equivalent square section for the root and tip properties
        root_A = surface.root_section.area
        root_Iyy = surface.root_section.Iyy
        root_Izz = surface.root_section.Izz
        root_J = surface.root_section.J
        root_E = surface.root_section.material.elasticity_modulus
        root_G = surface.root_section.materialrigidity_modulus

        root_A_sqside = np.sqrt(root_A)
        root_Iyy_sqside = (12 * root_Iyy) ** (1 / 4)
        root_Izz_sqside = (12 * root_Izz) ** (1 / 4)
        root_J_sqside = (6 * root_J) ** (1 / 4)

        tip_A = surface.tip_section.area
        tip_Iyy = surface.tip_section.Iyy
        tip_Izz = surface.tip_section.Izz
        tip_J = surface.tip_section.J
        tip_E = surface.root_section.material.elasticity_modulus
        tip_G = surface.root_section.materialrigidity_modulus

        tip_A_sqside = np.sqrt(tip_A)
        tip_Iyy_sqside = (12 * tip_Iyy) ** (1 / 4)
        tip_Izz_sqside = (12 * tip_Izz) ** (1 / 4)
        tip_J_sqside = (6 * tip_J) ** (1 / 4)

        # Linear interpolation of the equivalent square section for the node properties
        As = np.zeros(n_nodes)
        Iyys = np.zeros(n_nodes)
        Izzs = np.zeros(n_nodes)
        Js = np.zeros(n_nodes)
        Es = np.zeros(n_nodes)
        Gs = np.zeros(n_nodes)

        root_y = surface_nodes_list[0].y
        tip_y = surface_nodes_list[-1].y

        interpolation = (
            lambda root_prop, tip_prop, node_y: (
                (root_prop - tip_prop) / (root_y - tip_y)
            )
            * (node_y - tip_y)
            + tip_prop
        )

        # Calculation of the interpolated properties for each of the nodes
        for i, node in enumerate(surface_nodes_list):

            node_y = node.y
            A_sqside = interpolation(root_A_sqside, tip_A_sqside, node_y)
            Iyy_sqside = interpolation(root_Iyy_sqside, tip_Iyy_sqside, node_y)
            Izz_sqside = interpolation(root_Izz_sqside, tip_Izz_sqside, node_y)
            J_sqside = interpolation(root_J_sqside, tip_J_sqside, node_y)

            node_E = interpolation(root_E, tip_E, node_y)
            node_G = interpolation(root_G, tip_G, node_y)
            node_A = A_sqside ** 2
            node_Iyy = (Iyy_sqside ** 4) / 12
            node_Izz = (Izz_sqside ** 4) / 12
            node_J = (J_sqside ** 4) / 6

            As[i] = node_A
            Iyys[i] = node_Iyy
            Izzs[i] = node_Izz
            Js[i] = node_J
            Es[i] = node_E
            Gs[i] = node_G

        # Create FEM elements

        surface_elements = []

        for i in range(n_nodes - 1):

            if prop_choice == "ROOT":
                A = As[i]
                Iyy = Iyys[i]
                Izz = Izzs[i]
                J = Js[i]
                E = Es[i]
                G = Gs[i]

            elif prop_choice == "TIP":
                A = As[i + 1]
                Iyy = Iyys[i + 1]
                Izz = Izzs[i + 1]
                J = Js[i + 1]
                E = Es[i + 1]
                G = Gs[i + 1]

            elif prop_choice == "MIDDLE":
                A = (As[i] + As[i + 1]) / 2
                Iyy = (Iyys[i] + Iyys[i + 1]) / 2
                Izz = (Izzs[i] + Izzs[i + 1]) / 2
                J = (Js[i] + Js[i + 1]) / 2
                E = (Es[i] + Es[i + 1]) / 2
                G = (Gs[i] + Gs[i + 1]) / 2

            else:
                print("ERROR: Invalid prop_choice, options are 'ROOT', 'TIP' and 'MIDDLE'")

            element = o.EulerBeamElement(
                node_A=surface_nodes_list[i],
                node_B=surface_nodes_list[i],
                A=A,
                Iyy=Iyy,
                Izz=Izz,
                J=J,
                E=E,
                G=G,
                prop_choice=prop_choice,
            )

            surface_elements.append(element)

        macrosurface_elements.append(surface_elements)

    return macrosurface_elements

# ==================================================================================================

def generate_beam_fem_elements():
    pass
