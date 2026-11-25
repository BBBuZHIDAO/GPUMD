"""GPU regression: normal cutoff is independent of the trained NEP cutoff.
Run: IWP_TEST_NEP=/path/nep.txt python3 -m unittest discover -s tests/iwp -p test_iwp_normal_cutoff.py -v
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

class NormalCutoffTest(unittest.TestCase):
    def run_model(self, cutoff, periodic=False, wrong_wire=False, directed=False, probe=0.0, without_iwp=False, details=False, small_box=False, nep_extra=False, elements=("Mo", "S"), keyword="nep_iwp", native_nep=False, full_elements=None, strain_x=0.0):
        binary = Path(os.environ.get('IWP_TEST_GPUMD', ROOT / 'src/gpumd'))
        nep = Path(os.environ.get('IWP_TEST_NEP', ROOT.parent / 'nep.txt'))
        if not binary.is_file() or not nep.is_file():
            self.skipTest('Set IWP_TEST_GPUMD and IWP_TEST_NEP to GPU executable and Mo/S NEP.')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            shutil.copyfile(nep, p / 'nep.txt')
            (p / 'map.nep').write_text('0 1 nep.txt\n' + ('2\n0\n0\n' if wrong_wire or directed else '1\n0\n'))
            row = f'3 8 2 1 1 15 1 3 1 1 {cutoff} 4.1\n'
            if directed:
                row = f'8 2 20 1000 1000 15 1 3 1 1 {cutoff} 12\n'
            if without_iwp:
                row = f'8 2 20 0 0 15 1 3 0 1 {cutoff} 12\n'
            (p / 'iwp.txt').write_text((f'{keyword} 5 ' + ' '.join(full_elements) + '\n0\n' + row * 25) if full_elements else f'{keyword} 2 {elements[0]} {elements[1]}\n0\n' + row * 4)
            atoms = []
            for element, x in [(elements[0], 38 if periodic else 10), (elements[1], 2.5 if periodic else 14.5)]:
                for y, z in [(10, 10), (10, 12), (12, 10), (12, 12)]:
                    group = 1 if wrong_wire and element == elements[1] else 0
                    atoms.append(f'{element} {x} {y} {z} {group}')
            if full_elements:
                for index, element in enumerate(('Mo', 'W', 'Mo', 'W', 'S', 'Se', 'Te', 'S')):
                    values = atoms[index].split()
                    values[0] = element
                    atoms[index] = ' '.join(values)
            if directed:
                atoms.append('S 14.4 11 10.8 0')
                atoms += [f'{v[0]} {v[1]} {float(v[2]) + 8} {v[3]} 1' for v in [a.split() for a in atoms]]
            if nep_extra:
                atoms.append('Mo 13.2 10 10 0')
            if small_box:
                atoms = [' '.join([v[0], str(float(v[1]) - (7.1 if nep_extra else 8)), *v[2:]]) for v in [a.split() for a in atoms]]
            if probe:
                v = atoms[8].split()
                v[1] = str(float(v[1]) + probe)
                atoms[8] = ' '.join(v)
            if strain_x:
                atoms = [' '.join([v[0], str(float(v[1]) * (1 + strain_x)), *v[2:]]) for v in [a.split() for a in atoms]]
            (p / 'model.xyz').write_text(f'{len(atoms)}\nLattice="{(12 if small_box else 40) * (1 + strain_x)} 0 0 0 40 0 0 0 40" Properties=species:S:1:pos:R:3:group:I:1\n' + '\n'.join(atoms) + '\n')
            (p / 'run.in').write_text(('potential nep.txt\n' if native_nep else 'potential iwp.txt map.nep\n') + 'velocity 1\nensemble nve\ntime_step 0.000001\ndump_thermo 1\ndump_xyz 1 forces.xyz precision double force potential virial\nrun 1\n')
            command = [str(binary)]
            sanitizer = os.environ.get('IWP_COMPUTE_SANITIZER')
            if sanitizer:
                command = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '99', *command]
            result = subprocess.run(command, cwd=p, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('radial cutoff = 4 A', result.stdout)
            if directed or details:
                forces = [[float(v) for v in line.split()[4:7]] for line in (p / 'forces.xyz').read_text().splitlines()[2:]]
            if directed:
                residual = max(abs(sum(row[axis] for row in forces)) for axis in range(3))
                self.assertLess(residual, 1e-4, f'Nonreciprocal normal graph violates momentum conservation: {residual}')
            if details:
                data = (p / 'forces.xyz').read_text().splitlines()[2:]
                potential = sum(float(line.split()[7]) for line in data)
                if details == 'virial':
                    virials = [[float(value) for value in line.split()[8:17]] for line in data]
                    return result.stdout + result.stderr, potential, forces, virials
                return result.stdout + result.stderr, potential, forces
            return result.stdout + result.stderr

    def test_five_element_matrix_and_reordered_types(self):
        _, energy, forces = self.run_model(6, directed=True, details=True, full_elements=('Mo', 'W', 'S', 'Se', 'Te'))
        _, reordered_energy, reordered_forces = self.run_model(6, directed=True, details=True, full_elements=('Te', 'Se', 'S', 'W', 'Mo'))
        self.assertAlmostEqual(energy, reordered_energy, delta=2e-5)
        for actual_row, expected_row in zip(forces, reordered_forces):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, delta=1e-4)

    def test_iwp_virial_matches_energy_strain_derivative(self):
        def iwp_only(strain):
            _, efull, _, vfull = self.run_model(6, directed=True, details='virial', strain_x=strain)
            _, enep, _, vnep = self.run_model(6, directed=True, details='virial', strain_x=strain, without_iwp=True)
            return efull - enep, sum(row[0] for row in vfull) - sum(row[0] for row in vnep)
        h = 0.0005
        eplus, _ = iwp_only(h)
        eminus, _ = iwp_only(-h)
        _, virial = iwp_only(0)
        self.assertAlmostEqual(virial, -(eplus - eminus) / (2*h), delta=0.02)

    def test_all_six_metal_chalcogen_combinations(self):
        for metal in ('Mo', 'W'):
            for chalcogen in ('S', 'Se', 'Te'):
                with self.subTest(metal=metal, chalcogen=chalcogen):
                    self.assertNotIn('ERROR IWP normal', self.run_model(6, elements=(metal, chalcogen)))

    def test_keyword_alias_matches_original_keyword(self):
        _, original_energy, original_forces = self.run_model(6, directed=True, details=True)
        _, alias_energy, alias_forces = self.run_model(6, directed=True, details=True, keyword='nep_iwp_m6x6')
        self.assertAlmostEqual(original_energy, alias_energy, delta=2e-5)
        for original, alias in zip(original_forces, alias_forces):
            for actual, expected in zip(alias, original):
                self.assertAlmostEqual(actual, expected, delta=1e-4)

    def test_intrawire_nep_matches_native_5_9_1_nep(self):
        _, iwp_energy, iwp_forces = self.run_model(6, without_iwp=True, details=True, nep_extra=True)
        _, native_energy, native_forces = self.run_model(6, native_nep=True, details=True, nep_extra=True)
        self.assertAlmostEqual(iwp_energy, native_energy, delta=2e-5)
        for actual_row, expected_row in zip(iwp_forces, native_forces):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, delta=1e-4)

    def test_six_angstrom_normal_search_includes_atoms_beyond_nep_and_global_cutoffs(self):
        # All four unlike neighbors are 4.5--5.315 A away; NEP=4 and global=4.1.
        self.assertNotIn('ERROR IWP normal', self.run_model(6))

    def test_directed_nearest_four_graph_conserves_total_force(self):
        # Five S candidates for four Mo atoms: nearest-four edges are not reciprocal.
        self.assertNotIn('ERROR IWP normal', self.run_model(6, directed=True))

    def test_normal_derivative_force_matches_energy_finite_difference(self):
        def iwp_only(probe):
            _, efull, ffull = self.run_model(6, directed=True, probe=probe, details=True)
            _, enep, fnep = self.run_model(6, directed=True, probe=probe, without_iwp=True, details=True)
            return efull - enep, ffull[8][0] - fnep[8][0]
        # A 0.005 A step resolves single-precision IWP energy differences.
        h = 0.005
        eplus, _ = iwp_only(h)
        eminus, _ = iwp_only(-h)
        _, force = iwp_only(0)
        self.assertAlmostEqual(force, -(eplus - eminus) / (2 * h), delta=0.002)

    def test_small_periodic_box_does_not_count_same_atom_twice(self):
        # With Lx=12 and cutoff=5, four bins wrap +/-2 to the same bin.
        # Only three unlike atoms lie inside 5 A; duplicated bins must not supply a fourth.
        self.assertIn('ERROR IWP normal', self.run_model(5, small_box=True))

    def test_small_box_normal_search_preserves_nep_energy_and_forces(self):
        _, e_large, f_large = self.run_model(6, without_iwp=True, details=True, nep_extra=True)
        _, e_small, f_small = self.run_model(6, without_iwp=True, details=True, nep_extra=True, small_box=True)
        self.assertAlmostEqual(e_small, e_large, delta=2e-5)
        for small, large in zip(f_small, f_large):
            for actual, expected in zip(small, large):
                self.assertAlmostEqual(actual, expected, delta=1e-4)

    def test_four_angstrom_normal_cutoff_excludes_those_atoms(self):
        self.assertIn('ERROR IWP normal', self.run_model(4))

    def test_normal_search_applies_periodic_minimum_image(self):
        self.assertNotIn('ERROR IWP normal', self.run_model(6, periodic=True))

    def test_other_wire_atoms_cannot_supply_normal_neighbors(self):
        self.assertIn('ERROR IWP normal', self.run_model(6, wrong_wire=True))

if __name__ == '__main__':
    unittest.main()


