import numpy as np


def beam_3D_stiff(E, A, L, G, J, Iyy, Izz):
    """ Calculates the local stiffness matrix of a beam finite element.

    Args:
        A (float): Area of the section
        J (float): Polar moment of inertia of the section
        Iyy (float): Moment of inertia of the section in the Y axis
        Izz (float): Moment of inertia of the section in the Z axis
        E (float): Elastic modulus of the material
        G (float): Shear modulus of the material

    Returns:
        K (np.array(dtype=float)): local stiffness matrix of a beam element
    """

    K = np.zeros((12, 12))

    K[0][0] = E * A / L
    K[0][6] = -E * A / L
    K[6][0] = K[0][6]

    K[1][1] = 12 * E * Izz / L ** 3
    K[1][5] = 6 * E * Izz / L ** 2
    K[5][1] = K[1][5]
    K[1][7] = -12 * E * Izz / L ** 3
    K[7][1] = K[1][7]
    K[1][11] = 6 * E * Izz / L ** 2
    K[11][1] = K[1][11]

    K[2][2] = 12 * E * Iyy / L ** 3
    K[2][4] = -6 * E * Iyy / L ** 2
    K[4][2] = K[2][4]
    K[2][8] = -12 * E * Iyy / L ** 3
    K[8][2] = K[2][8]
    K[2][10] = -6 * E * Iyy / L ** 2
    K[10][2] = K[2][10]

    K[3][3] = G * J / L
    K[3][9] = -G * J / L
    K[9][3] = K[3][9]

    K[4][4] = 4 * E * Iyy / L
    K[4][8] = 6 * E * Iyy / L ** 2
    K[8][4] = K[4][8]
    K[4][10] = 2 * E * Iyy / L
    K[10][4] = K[4][10]

    K[5][5] = 4 * E * Izz / L
    K[5][7] = -6 * E * Izz / L ** 2
    K[7][5] = K[5][7]
    K[5][11] = 2 * E * Izz / L
    K[11][5] = K[5][11]

    K[6][6] = E * A / L

    K[7][7] = 12 * E * Izz / L ** 3
    K[7][11] = -6 * E * Izz / L ** 2
    K[11][7] = K[7][11]

    K[8][8] = 12 * E * Iyy / L ** 3
    K[8][10] = 6 * E * Iyy / L ** 2
    K[10][8] = K[8][10]

    K[9][9] = G * J / L

    K[10][10] = 4 * E * Iyy / L

    K[11][11] = 4 * E * Izz / L

    return K
