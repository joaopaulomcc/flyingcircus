"""
Aeroelasticity submodule
"""
import sys

import numpy as np

from pyquaternion import Quaternion

from .. import geometry as geo
from .. import aerodynamics as aero
from .. import structures as struct
from .. import mathematics as m

# ==================================================================================================


def loads_to_nodes_weight_matrix(
    macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
):

    node_vector = geo.functions.create_macrosurface_node_vector(
        macrosurface_struct_grid
    )
    panel_grid = aero.vlm.create_panel_grid(macrosurface_aero_grid)
    panel_vector = aero.vlm.flatten(panel_grid)

    weight_matrix = np.zeros((len(node_vector), len(panel_vector)))

    if algorithm == "closest":

        closest_node = None
        min_distance = float("inf")

        for j, panel in enumerate(panel_vector):

            for i, node in enumerate(node_vector):

                distance = geo.functions.distance_between_points(
                    panel.aero_center, node.xyz
                )

                if distance <= min_distance:

                    closest_node_index = i
                    min_distance = distance

            weight_matrix[closest_node_index][j] = 1

            closest_node = None
            min_distance = float("inf")

        return weight_matrix


# ==================================================================================================


def deformation_to_aero_grid_weight_matrix(
    macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
):

    node_vector = geo.functions.create_macrosurface_node_vector(
        macrosurface_struct_grid
    )

    aero_grid = geo.functions.macrosurface_aero_grid_to_single_grid(
        macrosurface_aero_grid
    )

    aero_points_vector = geo.functions.grid_to_vector(
        aero_grid["xx"], aero_grid["yy"], aero_grid["zz"]
    ).transpose()

    weight_matrix = np.zeros((len(aero_points_vector), len(node_vector)))

    if algorithm == "closest":

        closest_node = None
        min_distance = float("inf")

        for i, point in enumerate(aero_points_vector):

            for j, node in enumerate(node_vector):

                distance = geo.functions.distance_between_points(point, node.xyz)

                if distance <= min_distance:

                    closest_node_index = j
                    min_distance = distance

            weight_matrix[i][closest_node_index] = 1

            closest_node = None
            min_distance = float("inf")

        return weight_matrix


# ==================================================================================================


def generated_aero_loads(
    macrosurface_aero_grid,
    macro_surface_force_grid,
    macrosurface_struct_grid,
    algorithm="closest",
):

    macro_surface_loads = []

    node_vector = geo.functions.create_macrosurface_node_vector(
        macrosurface_struct_grid
    )
    panel_grid = aero.vlm.create_panel_grid(macrosurface_aero_grid)
    panel_vector = aero.vlm.flatten(panel_grid)
    force_vector = aero.vlm.flatten(macro_surface_force_grid)

    # Changes force_vector from array of arrays to a single numpy array
    force_vector = np.stack(force_vector)

    weight_matrix = loads_to_nodes_weight_matrix(
        macrosurface_aero_grid, macrosurface_struct_grid, algorithm=algorithm
    )

    for i, node_line in enumerate(weight_matrix):

        node = node_vector[i]

        node_force = np.zeros(3)
        node_moment = np.zeros(3)

        for j, panel_weight in enumerate(node_line):

            panel = panel_vector[j]

            force = force_vector[j] * panel_weight
            r = panel.aero_center - node.xyz
            moment = m.cross(r, force)

            node_force += force
            node_moment += moment

        load_components = np.array(
            [
                node_force[0],
                node_force[1],
                node_force[2],
                node_moment[0],
                node_moment[1],
                node_moment[2],
            ]
        )

        load = struct.objects.Load(application_node=node, load=load_components)

        macro_surface_loads.append(load)

    return macro_surface_loads


# ==================================================================================================


def deform_aero_grid(
    macrosurface_aero_grid,
    macrosurface_struct_grid,
    macrosurface_struct_deformations,
    algorithm="closest",
):

    node_vector = geo.functions.create_macrosurface_node_vector(
        macrosurface_struct_grid
    )

    surface_grids_shapes = []
    for surface_aero_grid in macrosurface_aero_grid:
        shape = np.shape(surface_aero_grid["xx"])
        surface_grids_shapes.append(shape)

    aero_grid = geo.functions.macrosurface_aero_grid_to_single_grid(
        macrosurface_aero_grid
    )

    aero_points_vector = geo.functions.grid_to_vector(
        aero_grid["xx"], aero_grid["yy"], aero_grid["zz"]
    ).transpose()

    weight_matrix = deformation_to_aero_grid_weight_matrix(
        macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
    )

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    deformed_points_vector = np.zeros(np.shape(aero_points_vector))

    for i, point_line in enumerate(weight_matrix):

        # Use Node Object to rotate and translate point

        point = aero_points_vector[i]
        point_node_object = geo.objects.Node(point, Quaternion())

        for j, node_weight in enumerate(point_line):

            node = node_vector[j]
            deformation = macrosurface_struct_deformations[j]

            # Apply rotations to point in relation to its correspondent structural node
            x_rot_quat = Quaternion(axis=x_axis, angle=(deformation[3] * node_weight))
            y_rot_quat = Quaternion(axis=y_axis, angle=(deformation[4] * node_weight))
            z_rot_quat = Quaternion(axis=z_axis, angle=(deformation[5] * node_weight))

            # X -> Y -> Z rotation
            # very small rotations are commutative, kind of
            point_node_object = point_node_object.rotate(x_rot_quat, node.xyz)
            point_node_object = point_node_object.rotate(y_rot_quat, node.xyz)
            point_node_object = point_node_object.rotate(z_rot_quat, node.xyz)

            # Apply translation
            translation_vector = deformation[:3] * node_weight
            point_node_object = point_node_object.translate(translation_vector)

        deformed_points_vector[i][0] = point_node_object.x
        deformed_points_vector[i][1] = point_node_object.y
        deformed_points_vector[i][2] = point_node_object.z

    x_grid, y_grid, z_grid = geo.functions.vector_to_grid(
        deformed_points_vector.transpose(), np.shape(aero_grid["xx"])
    )

    deformed_macrosurface_single_aero_grid = {"xx": x_grid, "yy": y_grid, "zz": z_grid}

    deformed_macrosurface_aero_grid = []

    slices = []
    slice_ends = 0
    for i in range(len(surface_grids_shapes) - 1):
        span_n_points = surface_grids_shapes[i][1]
        slice_ends += span_n_points
        slices.append(slice_ends)

    x_grids = np.split(deformed_macrosurface_single_aero_grid["xx"], slices, axis=1)
    y_grids = np.split(deformed_macrosurface_single_aero_grid["yy"], slices, axis=1)
    z_grids = np.split(deformed_macrosurface_single_aero_grid["zz"], slices, axis=1)

    for x_grid, y_grid, z_grid in zip(x_grids, y_grids, z_grids):

        deformed_macrosurface_aero_grid.append(
            {"xx": x_grid, "yy": y_grid, "zz": z_grid}
        )

    return deformed_macrosurface_aero_grid


# ==================================================================================================


def generate_aircraft_grids(aircraft_object, macrosurfaces_grid_data, beams_grid_data):

    macrosurfaces_aero_grids = []
    macrosurfaces_struct_grids = []
    macrosurfaces_connections = []
    macrosurfaces_components = []

    beams_struct_grids = []

    aircraft_components_list = []
    aircraft_components_nodes_list = []
    aircraft_connections_list = []

    # Create aerodynamic and structural grids for the aircraft macrosurfaces
    if aircraft_object.macrosurfaces:

        for i, macrosurface in enumerate(aircraft_object.macrosurfaces):

            grid_data = macrosurfaces_grid_data[i]

            macrosurface_aero_grid, macrosurface_struct_grid = macrosurface.create_grids(
                n_chord_panels=grid_data["n_chord_panels"],
                n_span_panels_list=grid_data["n_span_panels_list"],
                n_beam_elements_list=grid_data["n_beam_elements_list"],
                chord_discretization=grid_data["chord_discretization"],
                span_discretization_list=grid_data["span_discretization_list"],
                torsion_function_list=grid_data["torsion_function_list"],
                control_surface_deflection_dict=grid_data[
                    "control_surface_deflection_dict"
                ],
            )

            macrosurfaces_aero_grids.append(macrosurface_aero_grid)
            macrosurfaces_struct_grids.append(macrosurface_struct_grid)

            # Create connections for the macrosurface surfaces
            macrosurface_connections = struct.fem.create_macrosurface_connections(
                macrosurface
            )
            macrosurfaces_connections.append(macrosurface_connections)

            # Add macrosurface's surfaces to a single vector
            macrosurfaces_components.append(macrosurface.surface_list)

    else:

        print(
            "geometry.function.generate_aircraft_grids: ERROR No macrosurfaces were found"
        )
        print("Quitting execution...")
        sys.exit()

    # Add each macrosurface's surfaces, it's nodes, and connections to a single list
    for (
        macrosurface_surfaces,
        macrosurface_surfaces_nodes,
        macrosurface_connections,
    ) in zip(
        macrosurfaces_components, macrosurfaces_struct_grids, macrosurfaces_connections
    ):

        for connection in macrosurface_connections:
            aircraft_connections_list.append(connection)

        for surface, surface_nodes in zip(
            macrosurface_surfaces, macrosurface_surfaces_nodes
        ):

            aircraft_components_list.append(surface)
            aircraft_components_nodes_list.append(surface_nodes)

    # Structural grids for the aircraft beams
    if aircraft_object.beams:

        for i, beam in enumerate(aircraft_object.beams):

            grid_data = beams_grid_data[i]
            beam_struct_grid = beam.create_grid(n_elements=grid_data["n_elements"])

        beams_struct_grids.append(beam_struct_grid)

        # Add aircraft's beams, and it's nodes to the aircraft components and nodes list
        for beam, beam_nodes_list in zip(aircraft_object.beams, beams_struct_grids):
            aircraft_components_list.append(beam)
            aircraft_components_nodes_list.append(beam_nodes_list)

    # Add the structural connections to the aircraft connections list
    if aircraft_object.connections:
        for connection in aircraft_object.connections:
            aircraft_connections_list.append(connection)

    # Number the aircraft structural nodes
    struct.fem.number_nodes(
        aircraft_components_list,
        aircraft_components_nodes_list,
        aircraft_connections_list,
    )

    if aircraft_object.beams:

        aircraft_grids =  {
            "macrosurfaces_aero_grids": macrosurfaces_aero_grids,
            "macrosurfaces_struct_grids": macrosurfaces_struct_grids,
            "beams_struct_grids": beams_struct_grids,
        }

    else:

        aircraft_grids = {
            "macrosurfaces_aero_grids": macrosurfaces_aero_grids,
            "macrosurfaces_struct_grids": macrosurfaces_struct_grids,
            "beams_struct_grids": None,
        }

    return aircraft_grids

# ==================================================================================================


def generate_aircraft_constraints(aircraft, aircraft_grids, constraints_data_list):

    aircraft_constraints = []

    for constraint_data in constraints_data_list:

        # Run trought all surfaces in the aircraft searching for the identifier in each
        # of the constraints described in constraints_data_list

        for i, macrosurface in enumerate(aircraft.macrosurfaces):

            for j, surface in enumerate(macrosurface.surface_list):

                if surface.identifier == constraint_data["component_identifier"]:

                    surface_grid = aircraft_grids["macrosurfaces_struct_grids"][i][j]

                    if constraint_data["fixation_point"] == "ROOT":

                        # If root select root node
                        constraint = struct.objects.Constraint(
                            application_node=surface_grid[0],
                            dof_constraints=constraint_data["dof_constraints"],
                        )

                    elif constraint_data["fixation_point"] == "TIP":

                        # If tip select tip node
                        constraint = struct.objects.Constraint(
                            application_node=surface_grid[-1],
                            dof_constraints=constraint_data["dof_constraints"],
                        )

                    aircraft_constraints.append(constraint)

        # Run trought all beams in the aircraft searching for the identifier in each
        # of the constraints described in constraints_data_list

        if aircraft.beams:

            for i, beam in enumerate(aircraft.beams):

                if beam.identifier == constraint_data["component_identifier"]:

                    beam_grid = aircraft_grids["beams_struct_grids"][i]

                    if constraint_data["fixation_point"] == "ROOT":

                        # If root select root node
                        constraint = struct.objects.Constraint(
                            application_node=beam_grid[0],
                            dof_constraints=constraint_data["dof_constraints"],
                        )

                    elif constraint_data["fixation_point"] == "TIP":

                        # If tip select tip node
                        constraint = struct.objects.Constraint(
                            application_node=beam_grid[-1],
                            dof_constraints=constraint_data["dof_constraints"],
                        )

                    aircraft_constraints.append(constraint)

    return aircraft_constraints
