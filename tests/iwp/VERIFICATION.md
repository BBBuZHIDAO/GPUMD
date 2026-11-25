# GPUMD 5.9.1 IWP migration verification

- Independent `IWP_NEP_M6X6` CUDA implementation and embedded radial normal helper; no `iwp_normal.cuh`/`ilp_nep.cuh` dependency.
- Allowed elements: Mo W S Se Te. Metals +1, chalcogens -1. Five-element row-major template has 25 parameter rows.
- Old IWP functional form, four nearest same-wire M-X normals, separate normal cutoff, analytical derivatives and directed force/virial scatter retained.
- 5.9.1 NEP4 parameter layout and const group dispatch retained.
- Full CUDA12.9 sm120 compile and link: exit 0.
- CPU normal Jacobian/element signs, parameter converter and energy decomposition: pass.
- 13 GPU tests: pass, zero skips. Includes all 6 M-X combinations, 5x5 matrix/reordered elements, native NEP4 energy/force parity, force finite differences and virial strain derivatives.
- Independent code review: one inherited tail-thread guard defect fixed in new module; follow-up review found no remaining blockers.
- Original 5.9.1 archive file comparison: only src/force/force.cu and doc/potentials/index.rst changed; original ilp_nep.cu/.cuh unchanged byte for byte.

Limitations: W/Te fitted IWP coefficients were not provided, so template placeholders must be filled with fitted values. Imported 3-element example only has four fitted pair rows and limited-use fallback rows. NEP3/NEP5 legacy extensions were not ported; this module follows 5.9.1's NEP4 support. CUDA sm>=60 required. HIP and replica IWP not tested. Compute Sanitizer could not initialize the WSL/WDDM debugger interface and did not produce a valid memory-check result.
