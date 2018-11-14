import numpy as np
import matplotlib.pyplot as plt

from context import src
from src import aerodynamics
from src import geometry
from src import visualization

# ==================================================================================================
# FUNCTIONS

surface_identifier = "right_aileron"
root_chord = 2
root_section = "root_section"
tip_chord = 2
tip_section = "tip_section"
length = 3
leading_edge_sweep_angle_deg = 0
dihedral_angle_deg = 0
tip_torsion_angle_deg = 0
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
control_surface_deflection_dict = {"left_aileron": -15, "right_aileron": 15}

position = np.array([0.0, 0, 0.0])
incidence = 0

wing = geometry.objects.MacroSurface(
    position, incidence, surface_list, symmetry_plane="XZ"
)

n_chord_panels = 5
n_span_panels_list = [10, 10]
chord_discretization = "linear"
span_discretization_list = ["linear", "linear"]
torsion_function_list = ["linear", "linear"]

wing_mesh = wing.create_mesh(
    n_chord_panels,
    n_span_panels_list,
    chord_discretization,
    span_discretization_list,
    torsion_function_list,
    control_surface_deflection_dict,
)


panel_grid = aerodynamics.vlm.create_panel_grid(wing_mesh)
panel_vector = aerodynamics.vlm.flatten(panel_grid)

i, j = np.shape(panel_grid)

true_airspeed = 100
alpha = 5
beta = 0
gamma = 0

flow_velocity_vector = geometry.functions.velocity_vector(true_airspeed, alpha, beta, gamma)

gamma_vector = aerodynamics.vlm.gamma_solver(panel_vector, flow_velocity_vector)
gamma_grid = np.reshape(gamma_vector, np.shape(panel_grid))

j = 0
results = []
for mesh in wing_mesh:
    n_chord, n_span = np.shape(mesh["xx"])
    slic = gamma_grid[:,j:j+n_span-1]
    results.append(slic)
    j += n_span -1

print()
visualization.plot_3D.plot_results(wing_mesh, results)
plt.show()
    

def test_create_panel_grid():

    panel_grid = aerodynamics.vlm.create_panel_grid(wing_mesh)

    for i in range(np.shape(panel_grid)[0]):
        print(f"# i = {i}")
        for j in range(np.shape(panel_grid)[1]):
            print(panel_grid[i][j].induced_velocity(np.array([0, 0, 0]), 1))


def test_flatten():

    panel_vector = aerodynamics.vlm.flatten(panel_grid)

    for panel in panel_vector:
        print(panel.induced_velocity(np.array([0, 0, 0]), 1))

    re_panel_grid = np.reshape(panel_vector, np.shape(panel_grid))

    for i in range(np.shape(re_panel_grid)[0]):
        print(f"# i = {i}")
        for j in range(np.shape(re_panel_grid)[1]):
            print(re_panel_grid[i][j].induced_velocity(np.array([0, 0, 0]), 1))


def test_gamma_solver():

    gamma_vector = aerodynamics.vlm.gamma_solver(panel_vector, flow_velocity_vector)
    print("Gamma Vector:")
    for i, gamma in enumerate(gamma_vector):
        print(f"{i} : {gamma}")
        
    gamma_grid = np.reshape(gamma_vector, np.shape(panel_grid))
    visualization.plot_3D.plot_results(wing_mesh, gamma_grid)
    plt.show()


# ==================================================================================================
# TESTS

if __name__ == "__main__":

    print()
    print("============================")
    print("= Testing aerodynamics.vlm =")
    print("============================")
    print()

    #print("- Testing create_panel_grid")
    #test_create_panel_grid()
    #print()

    #print("- Testing flatten")
    #test_flatten()
    #print()

    #print("- Testing gamma_solver")
    #test_gamma_solver()
    #print()
