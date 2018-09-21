from context import src

from src import objects as obj

def test_wing():

    area = 20
    aspect_ratio = 5
    taper_ratio = 1
    sweep_quarter_chord = 0
    dihedral = 0
    incidence = 0
    torsion = 0
    position = [0, 0, 0]

    simple_rectangular = obj.Wing(area, aspect_ratio, taper_ratio, sweep_quarter_chord, dihedral,
                                 incidence, torsion, position)

    print(f"area: {simple_rectangular.area}")
    print(f"AR: {simple_rectangular.AR}")
    print(f"taper_ratio: {simple_rectangular.taper_ratio}")
    print(f"sweep: {simple_rectangular.sweep}")
    print(f"sweep_rad: {simple_rectangular.sweep_rad}")
    print(f"dihedral: {simple_rectangular.dihedral}")
    print(f"dihedral_rad: {simple_rectangular.dihedral_rad}")
    print(f"incidence: {simple_rectangular.incidence}")
    print(f"incidence_rad: {simple_rectangular.incidence_rad}")
    print(f"torsion: {simple_rectangular.torsion}")
    print(f"torsion_rad: {simple_rectangular.torsion_rad}")
    print(f"position: {simple_rectangular.position}")
    print(f"wing_span: {simple_rectangular.wing_span}")
    print(f"semi_wing_span: {simple_rectangular.semi_wing_span}")
    print(f"root_chord: {simple_rectangular.root_chord}")
    print(f"tip_chord: {simple_rectangular.tip_chord}")


if __name__ == "__main__":

    print()
    print("============================")
    print("= Testing objects module =")
    print("============================")
    print()
    test_wing()