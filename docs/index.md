# Introduction

Flying Circus is a Python framework for the simulation and analysis of aerodynamic loads in flexible aircraft. Its inital development was concluded as part of my Masters Dissertation in aeronautical enginnering at [ITA][1]

The code is still poorly documented, I intend to make improvements in the following months. 

# Methodology

Flying Circus uses a simplified beam structure to represent the aircraft structure and a [Vortex Lattice Method][2] to calculate aerodynamic loads. The structure deformation is calculated using a [Finite Element Method][3] based on [Euler-Bernoulli Beam][4] elements and an interactive approach is used for the aerodynamic/structural coupling.

For now Flying Circus is only able to consider effects of static aeroelasticity, in the future I would like to expand it's capabilities in order to allow the analysis of dynamic aeroelastic phenomena.

[1]: http://www.ita.br/
[2]: https://en.wikipedia.org/wiki/Vortex_lattice_method
[3]: https://en.wikipedia.org/wiki/Finite_element_method
[4]: https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory

