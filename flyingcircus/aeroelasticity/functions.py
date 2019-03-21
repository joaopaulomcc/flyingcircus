"""
Aeroelasticity submodule
"""
import sys
import datetime
import time

import numpy as np

from pyquaternion import Quaternion
from numba import jit

from .. import geometry as geo
from .. import aerodynamics as aero
from .. import structures as struct
from .. import mathematics as m
from .. import visualization as vis

# ==================================================================================================

@jit
def calculate_loads_to_nodes_weight_matrix(
    macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
):

    node_vector = geo.functions.create_structure_node_vector(macrosurface_struct_grid)
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

@jit
def calculate_deformation_to_aero_grid_weight_matrix(
    macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
):

    node_vector = geo.functions.create_structure_node_vector(macrosurface_struct_grid)

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

@jit
def generated_aero_loads(
    macrosurface_aero_grid,
    macrosurface_force_grid,
    macrosurface_struct_grid,
    algorithm="closest",
    weight_matrix=None,
):

    macrosurface_loads = []

    node_vector = geo.functions.create_structure_node_vector(macrosurface_struct_grid)
    panel_grid = aero.vlm.create_panel_grid(macrosurface_aero_grid)
    panel_vector = aero.vlm.flatten(panel_grid)
    force_vector = aero.vlm.flatten(macrosurface_force_grid)

    # Changes force_vector from array of arrays to a single numpy array
    force_vector = np.stack(force_vector)

    if weight_matrix is None:

        weight_matrix = calculate_loads_to_nodes_weight_matrix(
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

        macrosurface_loads.append(load)

    return macrosurface_loads


# ==================================================================================================

@jit
def deform_aero_grid(
    macrosurface_aero_grid,
    macrosurface_struct_grid,
    struct_deformations,
    algorithm="closest",
    weight_matrix=None,
):

    node_vector = geo.functions.create_structure_node_vector(macrosurface_struct_grid)

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

    if weight_matrix is None:

        weight_matrix = calculate_deformation_to_aero_grid_weight_matrix(
            macrosurface_aero_grid, macrosurface_struct_grid, algorithm="closest"
        )

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    deformed_points_vector = np.zeros(np.shape(aero_points_vector))

    macrosurface_struct_deformations = []

    for node in node_vector:
        node_deformation = struct_deformations[node.number]
        macrosurface_struct_deformations.append(node_deformation)

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

@jit
def generate_aircraft_grids(aircraft_object, aircraft_grid_data):

    macrosurfaces_aero_grids = []
    macrosurfaces_struct_grids = []
    macrosurfaces_connections = []
    macrosurfaces_components = []

    beams_struct_grids = []

    aircraft_components_list = []
    aircraft_components_nodes_list = []
    aircraft_connections_list = []

    macrosurfaces_grid_data = aircraft_grid_data["macrosurfaces_grid_data"]
    beams_grid_data = aircraft_grid_data["beams_grid_data"]

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

        aircraft_grids = {
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

@jit
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


# ==================================================================================================


def calculate_aircraft_loads(
    aircraft_object,
    aircraft_grid_data,
    flight_condition_data,
    simulation_options={
        "flexible_aircraft": True,
        "status_messages": True,
        "max_iterations": 50,
        "bending_convergence_criteria": 0.01,
        "torsion_convergence_criteria": 0.01,
        "fem_prop_choice": "ROOT",
        "interaction_algorithm": "closest",
        "output_iteration_results": True,
    },
    aircraft_constraints_data=None,
    influence_coef_matrix=None,
):

    output_iter = simulation_options["output_iteration_results"]
    iteration_results = []

    status = simulation_options["status_messages"]
    simulation_start_time = time.time()

    if status:
        print("# Running simulation")

    # Generate aircraft grids
    if status:
        print("- Generating aircraft grids ...")

    grid_start_time = time.time()

    macrosurfaces_grid_data = aircraft_grid_data["macrosurfaces_grid_data"]
    beams_grid_data = aircraft_grid_data["beams_grid_data"]

    aircraft_grids = generate_aircraft_grids(
        aircraft_object=aircraft_object, aircraft_grid_data=aircraft_grid_data
    )

    grid_end_time = time.time()

    aircraft_macrosurfaces_aero_grids = aircraft_grids["macrosurfaces_aero_grids"]
    aircraft_macrosurfaces_struct_grids = aircraft_grids["macrosurfaces_struct_grids"]
    aircraft_beams_struct_grids = aircraft_grids["beams_struct_grids"]

    control_node_string = simulation_options["control_node_string"]
    control_node_number = find_control_node_number(aircraft_object, aircraft_grids, control_node_string)

    if status:
        print(
            f"- Generating aircraft grids - Completed in {str(datetime.timedelta(seconds=(grid_end_time - grid_start_time)))}"
        )

    if simulation_options["flexible_aircraft"]:

        # Generate aircraft FEM elements

        fem_start_time = time.time()

        if status:
            print(f"- Generating aircraft FEM Elements ...")

        fem_prop_choice = simulation_options["fem_prop_choice"]

        aircraft_fem_elements = struct.fem.generate_aircraft_fem_elements(
            aircraft=aircraft_object,
            aircraft_grids=aircraft_grids,
            prop_choice=fem_prop_choice,
        )

        aircraft_macrosurfaces_fem_elements = aircraft_fem_elements[
            "macrosurfaces_fem_elements"
        ]
        aircraft_beams_fem_elements = aircraft_fem_elements["beams_fem_elements"]

        fem_end_time = time.time()

        if status:
            print(
                f"- Generating aircraft FEM Elements - Completed in {str(datetime.timedelta(seconds=(fem_end_time - fem_start_time)))}"
            )

        # Generate aircraft constraints

        aircraft_constraints = generate_aircraft_constraints(
            aircraft=aircraft_object,
            aircraft_grids=aircraft_grids,
            constraints_data_list=aircraft_constraints_data,
        )

        # Generate fluid/structure interaction matrices

        if status:
            print(f"- Generating aircraft fluid/structure interaction matrices ...")

        matrix_start_time = time.time()

        loads_to_nodes_macrosurfaces_weight_matrices = []
        deformation_to_aero_grid_macrosurfaces_weight_matrices = []
        interaction_algorithm = simulation_options["interaction_algorithm"]

        for i, macrosurface in enumerate(aircraft_object.macrosurfaces):

            macrosurface_aero_grid = aircraft_macrosurfaces_aero_grids[i]
            macrosurface_struct_grid = aircraft_macrosurfaces_struct_grids[i]

            loads_to_nodes_matrix = calculate_loads_to_nodes_weight_matrix(
                macrosurface_aero_grid, macrosurface_struct_grid, interaction_algorithm
            )

            deformation_to_aero_grid_weight_matrix = calculate_deformation_to_aero_grid_weight_matrix(
                macrosurface_aero_grid, macrosurface_struct_grid, interaction_algorithm
            )

            loads_to_nodes_macrosurfaces_weight_matrices.append(loads_to_nodes_matrix)
            deformation_to_aero_grid_macrosurfaces_weight_matrices.append(
                deformation_to_aero_grid_weight_matrix
            )

        matrix_end_time = time.time()

        if status:
            print(
                f"- Generating aircraft fluid/structure interaction matrices - Completed in {str(datetime.timedelta(seconds=(matrix_end_time - matrix_start_time)))}"
            )

        # Aeroelastic Calculation Loop

        aelast_loop_start_time = time.time()

        if status:
            print(f"- Running aeroelastic calculation ...")

        iteration_number = 0
        bending_delta = float("inf")
        torsion_delta = float("inf")

        max_iterations = simulation_options["max_iterations"]
        bending_convergence_criteria = simulation_options[
            "bending_convergence_criteria"
        ]
        torsion_convergence_criteria = simulation_options[
            "torsion_convergence_criteria"
        ]

        old_deformation = np.array([0, 0, 0, 0, 0, 0])

        while (bending_delta > bending_convergence_criteria) or (
            torsion_delta > torsion_convergence_criteria
        ):

            if iteration_number >= max_iterations:
                if status:
                    print(
                        f"    - Maximum number of iterarions, {max_iterations}, reached."
                    )
                break

            else:
                iteration_number += 1

            iteration_start_time = time.time()

            if status:
                print(f"    - Iteration {iteration_number}")

            # Calculate aerodynamic loads

            if iteration_number == 1:
                aircraft_deformed_macrosurfaces_aero_grids = (
                    aircraft_macrosurfaces_aero_grids
                )

            aero_start_time = time.time()
            (
                aircraft_force_vector,
                aircraft_panel_vector,
                aircraft_gamma_vector,
                aircraft_force_grid,
                aircraft_panel_grid,
                aircraft_gamma_grid,
                influence_coef_matrix,
            ) = aero.vlm.aero_loads(
                aircraft_aero_mesh=aircraft_deformed_macrosurfaces_aero_grids,
                velocity_vector=flight_condition_data["translation_velocity"],
                rotation_vector=flight_condition_data["rotation_velocity"],
                attitude_vector=flight_condition_data["attitude_angles_deg"],
                altitude=flight_condition_data["altitude"],
                center=flight_condition_data["center_of_rotation"],
                influence_coef_matrix=influence_coef_matrix,
            )
            aero_end_time = time.time()

            if iteration_number == 1:
                original_aircraft_panel_grid = aircraft_panel_grid

            if status:
                print(
                    f"        . Aerodynamic calculation completed in {str(datetime.timedelta(seconds=(aero_end_time - aero_start_time)))}"
                )

            # Calculate structure deformation

            struct_start_time = time.time()

            # Calculate aerodynamic loads in the structure
            aircraft_macrosurfaces_aero_loads = []

            for (
                macrosurface_aero_grid,
                macrosurface_force_grid,
                macrosurface_struct_grid,
                loads_to_nodes_weight_matrix,
            ) in zip(
                aircraft_macrosurfaces_aero_grids,
                aircraft_force_grid,
                aircraft_macrosurfaces_struct_grids,
                loads_to_nodes_macrosurfaces_weight_matrices,
            ):

                macrosurface_aero_loads = generated_aero_loads(
                    macrosurface_aero_grid,
                    macrosurface_force_grid,
                    macrosurface_struct_grid,
                    interaction_algorithm,
                    loads_to_nodes_weight_matrix,
                )

                aircraft_macrosurfaces_aero_loads.append(macrosurface_aero_loads)

            # Prepare input for structural solver

            struct_grid = []
            struct_fem_elements = []
            struct_loads = []
            struct_constraints = []

            # Add all surfaces grids to a vector
            for macrosurface_struct_grid in aircraft_macrosurfaces_struct_grids:

                struct_grid.extend(macrosurface_struct_grid)

            for macrosurface_fem_elements in aircraft_macrosurfaces_fem_elements:

                struct_fem_elements.extend(macrosurface_fem_elements)

            # Add all beams to a vector
            if aircraft_object.beams:

                struct_grid.extend(aircraft_beams_struct_grids)

                struct_fem_elements.extend(aircraft_beams_fem_elements)

            # Add all loads to a vector
            for macrosurface_aero_loads in aircraft_macrosurfaces_aero_loads:

                struct_loads.extend(macrosurface_aero_loads)

            # Add constraints to a vector
            struct_constraints.extend(aircraft_constraints)

            # Calculate structure deformations
            deformations, internal_loads = struct.fem.structural_solver(
                struct_grid, struct_fem_elements, struct_loads, struct_constraints
            )

            struct_end_time = time.time()
            if status:
                print(
                    f"        . Structural calculation completed in {str(datetime.timedelta(seconds=(struct_end_time - struct_start_time)))}"
                )

            # Deform aerodynamic grid

            def_start_time = time.time()

            aircraft_deformed_macrosurfaces_aero_grids = []
            aircraft_deformed_macrosurfaces_aero_panels = []

            for (
                macrosurface_struct_grid,
                macrosurface_aero_grid,
                deformation_to_aero_grid_weight_matrix,
            ) in zip(
                aircraft_macrosurfaces_struct_grids,
                aircraft_macrosurfaces_aero_grids,
                deformation_to_aero_grid_macrosurfaces_weight_matrices,
            ):

                deformed_macrosurface_aero_grid = deform_aero_grid(
                    macrosurface_aero_grid,
                    macrosurface_struct_grid,
                    deformations,
                    weight_matrix=deformation_to_aero_grid_weight_matrix,
                )

                deformed_macrosurface_aero_panels = aero.vlm.create_panel_grid(
                    deformed_macrosurface_aero_grid
                )

                aircraft_deformed_macrosurfaces_aero_grids.append(
                    deformed_macrosurface_aero_grid
                )
                aircraft_deformed_macrosurfaces_aero_panels.append(
                    deformed_macrosurface_aero_panels
                )

            def_end_time = time.time()
            if status:
                print(
                    f"        . Aerodynamic Grid deformation completed in {str(datetime.timedelta(seconds=(def_end_time - def_start_time)))}"
                )

            iteration_end_time = time.time()

            # Check convergence

            new_deformation = deformations[control_node_number]

            if np.array_equal(old_deformation, np.zeros([6])):
                delta_deformation = np.full((6), float("inf"))

            else:
                delta_deformation = np.abs(new_deformation - old_deformation) / np.abs(old_deformation)

            bending_delta = m.norm(delta_deformation[:3])
            torsion_delta = m.norm(delta_deformation[3:])

            old_deformation = np.copy(new_deformation)

            if output_iter:

                this_iteration_results = {
                    "iteration_number": iteration_number,
                    "aircraft_deformed_macrosurfaces_aero_grids": aircraft_deformed_macrosurfaces_aero_grids,
                    "aircraft_macrosurfaces_panel_grid": aircraft_deformed_macrosurfaces_aero_panels,
                    "aircraft_gamma_grid": aircraft_gamma_grid,
                    "aircraft_force_grid": aircraft_force_grid,
                    "aircraft_struct_deformations": deformations,
                    "aircraft_struct_internal_loads": internal_loads,
                    "deformation_at_control_node": old_deformation,
                    "influence_coef_matrix": influence_coef_matrix,
                }

                iteration_results.append(this_iteration_results)

            print(f"        . Bending Delta: {bending_delta}")
            print(f"        . Torsion Delta: {torsion_delta}")

            if status:
                print(
                    f"        . Iteration {iteration_number} completed in {str(datetime.timedelta(seconds=(iteration_end_time - iteration_start_time)))}"
                )
        aelast_loop_end_time = time.time()

        if status:
            print(
                f"- Running aeroelastic calculation - Completed in {str(datetime.timedelta(seconds=(aelast_loop_end_time - aelast_loop_start_time)))}"
            )

        simulation_end_time = time.time()
        if status:
            print(
                f"# Running simulation - Completed in {str(datetime.timedelta(seconds=(simulation_end_time - simulation_start_time)))}"
            )

        final_results = {
            "aircraft_deformed_macrosurfaces_aero_grids": aircraft_deformed_macrosurfaces_aero_grids,
            "aircraft_deformed_macrosurfaces_aero_panels": aircraft_deformed_macrosurfaces_aero_panels,
            "aircraft_gamma_grid": aircraft_gamma_grid,
            "aircraft_force_grid": aircraft_force_grid,
            "aircraft_struct_deformations": deformations,
            "aircraft_struct_internal_loads": internal_loads,
            "deformation_at_control_node": old_deformation,
            "influence_coef_matrix": influence_coef_matrix,
            "aircraft_original_grids": aircraft_grids,
            "aircraft_struct_fem_elements": aircraft_fem_elements,
            "original_aircraft_panel_grid": original_aircraft_panel_grid,
        }

        if output_iter:

            return final_results, iteration_results

        else:

            return final_results

    else:

        aero_start_time = time.time()

        (
            aircraft_force_vector,
            aircraft_panel_vector,
            aircraft_gamma_vector,
            aircraft_force_grid,
            aircraft_panel_grid,
            aircraft_gamma_grid,
            influence_coef_matrix,
        ) = aero.vlm.aero_loads(
            aircraft_aero_mesh=aircraft_macrosurfaces_aero_grids,
            velocity_vector=flight_condition_data["translation_velocity"],
            rotation_vector=flight_condition_data["rotation_velocity"],
            attitude_vector=flight_condition_data["attitude_angles_deg"],
            altitude=flight_condition_data["altitude"],
            center=flight_condition_data["center_of_rotation"],
            influence_coef_matrix=influence_coef_matrix,
        )

        aero_end_time = time.time()

        if status:
            print(
                f"        . Aerodynamic calculation completed in {str(datetime.timedelta(seconds=(aero_end_time - aero_start_time)))}"
            )

        results = {
            "aircraft_macrosurfaces_panels": aircraft_panel_grid,
            "aircraft_original_grids": aircraft_grids,
            "aircraft_gamma_grid": aircraft_gamma_grid,
            "aircraft_force_grid": aircraft_force_grid,
            "influence_coef_matrix": influence_coef_matrix,
        }

        return results

# ==================================================================================================

@jit
def find_control_node_number(aircraft, aircraft_grids, control_node_string):

    component_identifier, node_position = control_node_string.split("-")

    # Run trought all surfaces in the aircraft searching for the identifier in each
    # of the constraints described in constraints_data_list

    for i, macrosurface in enumerate(aircraft.macrosurfaces):

        for j, surface in enumerate(macrosurface.surface_list):

            if surface.identifier == component_identifier:

                surface_grid = aircraft_grids["macrosurfaces_struct_grids"][i][j]

                if node_position == "ROOT":

                    # If root select root node
                    node_number = surface_grid[0].number

                elif node_position == "TIP":

                    # If tip select tip node
                    node_number = surface_grid[-1].number

                return node_number

    # Run trought all beams in the aircraft searching for the identifier in each
    # of the constraints described in constraints_data_list

    if aircraft.beams:

        for i, beam in enumerate(aircraft.beams):

            if beam.identifier == component_identifier:

                beam_grid = aircraft_grids["beams_struct_grids"][i]

                if node_position == "ROOT":

                    # If root select root node
                    node_number = beam_grid[0],

                elif node_position == "TIP":

                    # If tip select tip node
                    node_number = beam_grid[-1],

                return node_number

    print("ERROR: Control Node was not found")
    return None


# ==================================================================================================

@jit
def calculate_deformation_table(aircraft_original_grids, aircraft_struct_deformations):

    aircraft_macrosurfaces_deformed_nodes = []

    for macrosurface_struct_grid in aircraft_original_grids["macrosurfaces_struct_grids"]:

        node_vector = geo.functions.create_structure_node_vector(macrosurface_struct_grid)

        deformed_nodes = np.zeros((len(node_vector), 6))

        for i, node in enumerate(node_vector):
            deformed_nodes[i][0] = node.x + aircraft_struct_deformations[node.number][0]
            deformed_nodes[i][1] = node.y + aircraft_struct_deformations[node.number][1]
            deformed_nodes[i][2] = node.z + aircraft_struct_deformations[node.number][2]
            deformed_nodes[i][3] = aircraft_struct_deformations[node.number][3]
            deformed_nodes[i][4] = aircraft_struct_deformations[node.number][4]
            deformed_nodes[i][5] = aircraft_struct_deformations[node.number][5]

        aircraft_macrosurfaces_deformed_nodes.append(deformed_nodes)

    aircraft_beams_deformed_nodes = []

    if aircraft_original_grids["beams_struct_grids"]:

        for beam in aircraft_original_grids["beams_struct_grids"]:

            deformed_nodes = np.zeros((len(beam), 6))

            for i, node in enumerate(beam):
                deformed_nodes[i][0] = node.x + aircraft_struct_deformations[node.number][0]
                deformed_nodes[i][1] = node.y + aircraft_struct_deformations[node.number][1]
                deformed_nodes[i][2] = node.z + aircraft_struct_deformations[node.number][2]
                deformed_nodes[i][3] = aircraft_struct_deformations[node.number][3]
                deformed_nodes[i][4] = aircraft_struct_deformations[node.number][4]
                deformed_nodes[i][5] = aircraft_struct_deformations[node.number][5]

            aircraft_beams_deformed_nodes.append(deformed_nodes)

    return {"aircraft_macrosurfaces_deformed_nodes": aircraft_macrosurfaces_deformed_nodes,
            "aircraft_beams_deformed_nodes": aircraft_beams_deformed_nodes}