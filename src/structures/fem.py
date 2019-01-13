import numpy as np
import scipy as sc

from numpy import sin, cos, tan, pi
from pyquaternion import Quaternion

from .. import geometry as geo
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

            # If the node is the root of the component check if it is connected to a node that has
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
            elif j == len(components_nodes_list[i]) - 1:
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

            if connected_component_index == 0:
                connected_component = connections_list[i].component1
                connected_node = connections_list[i].component1_node

            elif connected_component_index == 1:
                connected_component = connections_list[i].component2
                connected_node = connections_list[i].component2_node

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

    return connections_list


# ==================================================================================================


def generate_macrosurface_fem_elements(
    macrosurface, macrosurface_nodes_list, prop_choice="MIDDLE"
):
    """Generates the beam finite elements of a macrosurface object.

    Args

    Returns
    """

    macrosurface_elements = []

    for surface, surface_nodes_list in zip(
        macrosurface.surface_list, macrosurface_nodes_list
    ):

        n_nodes = len(surface_nodes_list)

        # Calculates the equivalent square section for the root and tip properties
        root_A = surface.root_section.area
        root_Iyy = surface.root_section.Iyy
        root_Izz = surface.root_section.Izz
        root_J = surface.root_section.J
        root_E = surface.root_section.material.elasticity_modulus
        root_G = surface.root_section.material.rigidity_modulus

        root_A_sqside = np.sqrt(root_A)
        root_Iyy_sqside = (12 * root_Iyy) ** (1 / 4)
        root_Izz_sqside = (12 * root_Izz) ** (1 / 4)
        root_J_sqside = (6 * root_J) ** (1 / 4)

        tip_A = surface.tip_section.area
        tip_Iyy = surface.tip_section.Iyy
        tip_Izz = surface.tip_section.Izz
        tip_J = surface.tip_section.J
        tip_E = surface.root_section.material.elasticity_modulus
        tip_G = surface.root_section.material.rigidity_modulus

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

        interpolation = lambda root_prop, tip_prop, node_y: (
            np.interp([node_y], [root_y, tip_y], [root_prop, tip_prop])
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
                print(
                    "ERROR: Invalid prop_choice, options are 'ROOT', 'TIP' and 'MIDDLE'"
                )

            element = o.EulerBeamElement(
                node_A=surface_nodes_list[i],
                node_B=surface_nodes_list[i + 1],
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


def generate_beam_fem_elements(beam, beam_nodes_list, prop_choice="MIDDLE"):

    n_nodes = len(beam_nodes_list)

    beam_elements = []

    for i in range(n_nodes - 1):

        A = beam.ElementProperty.A
        Iyy = beam.ElementProperty.Iyy
        Izz = beam.ElementProperty.Izz
        J = beam.ElementProperty.J
        E = beam.ElementProperty.E
        G = beam.ElementProperty.G

        element = o.EulerBeamElement(
            node_A=beam_nodes_list[i],
            node_B=beam_nodes_list[i + 1],
            A=A,
            Iyy=Iyy,
            Izz=Izz,
            J=J,
            E=E,
            G=G,
            prop_choice=prop_choice,
        )

        beam_elements.append(element)

    return beam_elements


# ==================================================================================================


def generate_aircraft_fem_elements(
    aircraft,
    aircraft_grids,
    prop_choice="ROOT",
):

    aircraft_macrosurfaces_struct_grids = aircraft_grids["macrosurfaces_struct_grids"]
    aircraft_beams_struct_grids = aircraft_grids["beams_struct_grids"]

    aircraft_macrosurfaces_fem_elements = []

    for macrosurface, macrosurface_struct_grid in zip(
        aircraft.macrosurfaces, aircraft_macrosurfaces_struct_grids
    ):

        macrosurface_fem_elements = generate_macrosurface_fem_elements(
            macrosurface, macrosurface_struct_grid, prop_choice
        )

        aircraft_macrosurfaces_fem_elements.append(macrosurface_fem_elements)

    aircraft_beams_fem_elements = []

    if aircraft.beams:
        for beam, beam_struct_grid in zip(aircraft.beams, aircraft_beams_struct_grids):

            beam_elements = generate_beam_fem_elements(
                beam, beam_struct_grid, prop_choice
            )
            aircraft_beams_fem_elements.append(beam_elements)

    if aircraft.beams:
        aircraft_fem_elements =  {
            "macrosurfaces_fem_elements":aircraft_macrosurfaces_fem_elements,
            "beams_fem_elements":aircraft_beams_fem_elements,
        }

    else:
        aircraft_fem_elements =  {
            "macrosurfaces_fem_elements":aircraft_macrosurfaces_fem_elements,
            "beams_fem_elements":None,
        }

    return aircraft_fem_elements

# ==================================================================================================


def structural_solver(struct_grid, struct_elements, struct_loads, struct_constraints):

    node_vector = geo.functions.create_macrosurface_node_vector(struct_grid)

    elements_vector = []
    # Add all elements to a vector
    for component_elements in struct_elements:
        elements_vector += component_elements

    K_global, F_global = create_global_FEM_matrices(
        node_vector, elements_vector, struct_loads
    )

    X_global = FEM_solver(K_global, F_global, struct_constraints)

    # Find support reactions
    force_vector = K_global @ X_global

    # Deformed grid
    deformations = np.reshape(X_global, (len(node_vector), 6))

    deformed_grid = []
    for i, node in enumerate(node_vector):
        deformed_grid.append(
            [
                node.x + deformations[i][0],
                node.y + deformations[i][1],
                node.z + deformations[i][2],
                deformations[i][3],
                deformations[i][4],
                deformations[i][5],
            ]
        )

    return deformed_grid, force_vector, deformations, node_vector


# ==================================================================================================


def create_global_FEM_matrices(nodes, fem_elements, loads):

    n_nodes = len(nodes)

    # Generate global stiffness matrix
    K_global = np.zeros((n_nodes * 6, n_nodes * 6))
    F_global = np.zeros((n_nodes * 6, 1))

    for fem_element in fem_elements:
        K_element = fem_element.calc_K_global()
        correlation_vector = fem_element.correlation_vector

        for i in range(len(correlation_vector)):
            for j in range(len(correlation_vector)):
                K_global[correlation_vector[i]][correlation_vector[j]] += K_element[i][
                    j
                ]

    # Generate Force Matrix
    for load in loads:
        node_index = load.application_node.number * 6
        correlation_vector = [
            node_index,
            node_index + 1,
            node_index + 2,
            node_index + 3,
            node_index + 4,
            node_index + 5,
        ]

        for i in range(len(correlation_vector)):
            F_global[correlation_vector[i]] += load.load[i]

    return K_global, F_global


# ==================================================================================================


def FEM_solver(K_global, F_global, constraints):

    n_dof = len(F_global)
    X_global = np.zeros((n_dof, 1))

    # Find constrained degrees of freedom
    constrained_dof = [False for i in range(n_dof)]

    for constraint in constraints:
        node_index = constraint.application_node.number * 6
        correlation_vector = [
            node_index,
            node_index + 1,
            node_index + 2,
            node_index + 3,
            node_index + 4,
            node_index + 5,
        ]

        for i in range(len(correlation_vector)):
            if constraint.dof_constraints[i] is not None:
                constrained_dof[correlation_vector[i]] = True
                X_global[correlation_vector[i]] += constraint.dof_constraints[i]

    # Created reduced stiffess and force matrices
    red_K_global = np.copy(K_global)
    red_F_global = np.copy(F_global)

    dof_to_delete = []
    for i, dof in enumerate(constrained_dof):
        if dof:
            dof_to_delete.append(i)

    red_F_global = np.delete(red_F_global, dof_to_delete, 0)
    red_K_global = np.delete(red_K_global, dof_to_delete, 0)
    red_K_global = np.delete(red_K_global, dof_to_delete, 1)

    # Solve linear System
    red_X_global = np.linalg.solve(red_K_global, red_F_global)

    # Copy results do deformation vector
    counter = 0
    for i, dof in enumerate(constrained_dof):
        if not dof:
            X_global[i] = red_X_global[counter]
            counter += 1

    return X_global
