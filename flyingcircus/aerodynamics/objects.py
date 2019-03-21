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
    
    def aero_force(self, circulation, flow_vector, air_density):
        hs_aero_force = functions.horse_shoe_aero_force(
            self.horse_shoe_point_a, self.horse_shoe_point_b, circulation, flow_vector, air_density
        )

        return hs_aero_force

# ==================================================================================================

