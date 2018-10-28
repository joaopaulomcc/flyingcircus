import numpy as np
import matplotlib.pyplot as plt
from context import src
from src import geometry
from src import visualization


def test_surface():

    surface_identifier = "right_aileron"
    root_chord = 2
    root_section = "root_section"
    tip_chord = 1.2
    tip_section = "tip_section"
    length = 3
    leading_edge_sweep_angle_deg = 35
    dihedral_angle_deg = 5
    tip_torsion_angle_deg = -5
    control_surface_hinge_position = 0.75

    right_aileron = geometry.objects.Surface(
        surface_identifier,
        root_chord,
        root_section,
        tip_chord,
        tip_section,
        length,
        leading_edge_sweep_angle_deg,
        dihedral_angle_deg,
        tip_torsion_angle_deg,
        control_surface_hinge_position,
    )

    surface_identifier = "left_aileron"

    left_aileron = geometry.objects.Surface(
        surface_identifier,
        root_chord,
        root_section,
        tip_chord,
        tip_section,
        length,
        leading_edge_sweep_angle_deg,
        dihedral_angle_deg,
        tip_torsion_angle_deg,
        control_surface_hinge_position,
    )

    surface_list = [left_aileron, right_aileron]
    control_surface_deflection_dict = {"left_aileron": -45, "right_aileron": 45}

    position = np.array([0.5, 0, 0.1])
    incidence = 5

    wing = geometry.objects.MacroSurface(
        position, incidence, surface_list, symmetry_plane="XZ"
    )

    n_span_panels = 20
    n_chord_panels = 20
    control_surface_deflection = 45
    chord_discretization = "cos_sim"
    span_discretization = "cos_sim"

    mesh_points_xx, mesh_points_yy, mesh_points_zz = right_aileron.generate_aero_mesh(
        n_span_panels,
        n_chord_panels,
        control_surface_deflection,
        chord_discretization,
        span_discretization,
    )

    wing_mesh = wing.create_mesh(
        n_chord_panels,
        n_span_panels,
        control_surface_deflection_dict,
        chord_discretization="linear",
        span_discretization="linear",
        torsion_function="linear",
    )

    visualization.plot_3D.plot_mesh([[mesh_points_xx, mesh_points_yy, mesh_points_zz]])
    visualization.plot_3D.plot_mesh(wing_mesh)

    plt.show()


if __name__ == "__main__":

    print()
    print("============================")
    print("= Testing geometry.objects =")
    print("============================")
    print()
    print("# Testing geometry.objects.Surface")
    test_surface()
