from . import functions
from .. import geometry as geo

# ==================================================================================================


class PanelHorseShoe(geo.objects.Panel):
    def __init__(self, xx, yy, zz):
        super().__init__(xx, yy, zz)

        self.horse_shoe_point_a = self.l_chord_1_4
        self.horse_shoe_point_b = self.r_chord_1_4

    def induced_velocity(self, target_point, circulation):
        hs_induced_velocity = functions.horse_shoe_ind_vel(
            self.horse_shoe_point_a, self.horse_shoe_point_b, target_point, circulation
        )

        return hs_induced_velocity

# ==================================================================================================

