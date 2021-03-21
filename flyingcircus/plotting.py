import numpy as np
import matplotlib.pyplot as plt

import vtk


def plot_3d_lines(x_vectors, y_vectors, z_vectors, colors=[1.0, 1.0, 1.0]):

    colors_shape = np.shape(colors)

    if len(x_vectors[0]) == 1:

        x_vectors = [x_vectors]
        y_vectors = [y_vectors]
        z_vectors = [z_vectors]

    actors = []

    for i, (x_vector, y_vector, z_vector) in enumerate(
        zip(x_vectors, y_vectors, z_vectors)
    ):

        mesh = create_vtk_line(x_vector, y_vector, z_vector)
        mesh_mapper = vtk.vtkPolyDataMapper()
        mesh_mapper.SetInputData(mesh)
        mesh_actor = vtk.vtkActor()
        mesh_actor.SetMapper(mesh_mapper)

        if len(colors_shape) == 1:
            mesh_actor.GetProperty().SetColor(colors)

        else:
            mesh_actor.GetProperty().SetColor(colors[i])

        actors.append(mesh_actor)

    renderer, interactor, window = create_vtk_scene(actors)

    interactor.Start()


def create_vtk_scene(
    actors,
    window_size=(1000, 800),
    window_title="Plot",
    window_background=(0.0, 0.0, 0.0),
):

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(window_background)

    for actor in actors:

        renderer.AddActor(actor)

    window = vtk.vtkRenderWindow()
    window.SetSize(window_size)
    window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    interactor.Initialize()
    renderer.ResetCamera()

    window.Render()
    window.SetWindowName(window_title)

    return renderer, interactor, window


def create_vtk_line(x_vector, y_vector, z_vector):

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    mesh = vtk.vtkPolyData()

    for x, y, z in zip(x_vector, y_vector, z_vector):

        points.InsertNextPoint(x, y, z)

    for i in range(len(x_vector) - 1):

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)

    mesh.SetPoints(points)
    mesh.SetLines(lines)

    return mesh


def plot_airfoil(name, x, y, show_points=False):

    fig, ax = plt.subplots()
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    # Hide the top and right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.plot(x, y)
    ax.set_title(name)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.axis("equal")

    if show_points:
        ax.scatter(x, y)

    plt.show()


