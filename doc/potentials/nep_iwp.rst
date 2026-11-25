NEP+IWP for M6X6 nanowires
==========================

The ``nep_iwp`` potential combines intrawire NEP models with an inter-wire
potential using the same pair functional form as ``nep_ilp``. Unlike planar
ILP normals, each IWP local normal is the normalized signed sum of the unit
vectors to the four nearest unlike atoms in the same wire.

Potential-file syntax::

  nep_iwp <number of atom types> <list of elements>
  <group method for wires>
  z0 alpha delta epsilon C d sR reff C6 S rcut_normal rcut_global
  ...

The group label identifies the wire. Interactions are evaluated only between
different wire labels, while normal neighbors are selected only within the
same wire label. Mo/W use normal sign +1 and S/Se/Te use -1. The numerical
file must contain ``N*N`` parameter rows and must not contain comment lines.

See ``README_IWP_M6X6.md`` and ``potentials/iwp`` for the Mo6S6/Mo6Se6
conversion example.

The eleventh parameter, ``rcut_normal`` (Angstrom), limits same-wire unlike-atom
normal candidates independently of the NEP descriptor cutoff and
``rcut_global``. The four nearest candidates within the pair-specific
``rcut_normal`` are selected. NEP descriptors retain their trained cutoffs.
Normal-neighbor derivative forces and virials are accumulated along directed
edges, so a reciprocal nearest-four graph is not required.
