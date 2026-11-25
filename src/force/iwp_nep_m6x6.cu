/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Independent M6X6 interwire potential (IWP) and neuroevolution 
potential(NEP).

Ref: Hekai Bu, Wenwu Jiang, Penghua Ying, Ting Liang, Zheyong Fan, and 
Wengen Ouyang, Modular hybrid machine learning and physics-based potentials 
for scalable modeling of Van der Waals heterostructures, 
J. Mech. Phys. Solids 210, 106540 (2026).

Implemented by: Hekai Bu (Wuhan University), hekai_bu@whu.edu.cn
------------------------------------------------------------------------------*/

// IWP radial normals and directed normal-force scatter migrated from
// GPUMD-4.0-nep-iwp-M6X6-v1; NEP4 kernels/parser follow GPUMD-5.9.1.
// Imported IWP code retains GPL-3.0-or-later licensing (LICENCE_IWP_GPL3).
#include "iwp_nep_m6x6.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_parameters.cuh"
#include "utilities/nep_utilities.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
// TODO
// #include <iomanip>


const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

IWP_NEP_M6X6::IWP_NEP_M6X6(FILE* fid_ilp, FILE* fid_nep_map, int num_types, int num_atoms,
  bool require_zbl)
{
  // Read IWP elements.
  printf("Use %d-element IWP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= IWP_M6X6_MAX_TYPE_ILP_NEP)) {
    PRINT_INPUT_ERROR("IWP requires 1 to 5 distinct types from Mo W S Se Te.\n");
  }
  std::vector<std::string> ilp_elements(num_types);
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid_ilp, "%9s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for IWP potential.");
    printf(" %s", atom_symbol);
    ilp_elements[n] = atom_symbol;
    iwp_normal_sign_cpu[n] = iwp_m6x6::normal_sign(atom_symbol);
    if (iwp_normal_sign_cpu[n] == 0.0f) {
      PRINT_INPUT_ERROR("IWP supports only Mo W S Se Te.\n");
    }
    for (int previous = 0; previous < n; ++previous) {
      if (ilp_elements[previous] == atom_symbol) {
        PRINT_INPUT_ERROR("Duplicate IWP element.\n");
      }
    }
    printf("%s", iwp_normal_sign_cpu[n] > 0 ? "(M,+)" : "(X,-)");
  }
  printf("\n");
  iwp_normal_sign_gpu.resize(IWP_M6X6_MAX_TYPE_ILP_NEP);
  iwp_normal_sign_gpu.copy_from_host(iwp_normal_sign_cpu);
  PRINT_SCANF_ERROR(fscanf(fid_ilp, "%d", &ilp_group_method), 1,
    "Reading error for IWP wire group method.");
  if (ilp_group_method < 0) PRINT_INPUT_ERROR("IWP group method must be non-negative.\n");
  printf("Use group method %d to identify wire for IWP.\n", ilp_group_method);

  // Read IWP pair parameters.
  float beta, alpha, delta, epsilon, C, d, sR;
  float reff, C6, S, rcut_ilp, rcut_global;
  rc = 0.0;
  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < num_types; ++m) {
      int count = fscanf(fid_ilp, "%f%f%f%f%f%f%f%f%f%f%f%f", \
      &beta, &alpha, &delta, &epsilon, &C, &d, &sR, &reff, &C6, &S, \
      &rcut_ilp, &rcut_global);
      PRINT_SCANF_ERROR(count, 12, "Reading error for IWP potential.");

      const float values[] = {beta, alpha, delta, epsilon, C, d, sR, reff, C6, S, rcut_ilp, rcut_global};
      for (float value : values) {
        if (!std::isfinite(value)) PRINT_INPUT_ERROR("IWP parameters must be finite.\n");
      }
      if (!(beta > 0 && delta > 0 && sR > 0 && reff > 0 && rcut_ilp > 0 && rcut_global > 0)) {
        PRINT_INPUT_ERROR("IWP z0, delta, sR, reff and cutoffs must be positive.\n");
      }
      max_iwp_normal_rc = std::max(max_iwp_normal_rc, double(rcut_ilp));
      ilp_para.C[n][m] = C;
      ilp_para.C_6[n][m] = C6;
      ilp_para.d[n][m] = d;
      ilp_para.d_Seff[n][m] = d / sR / reff;
      ilp_para.epsilon[n][m] = epsilon;
      ilp_para.z0[n][m] = beta;
      ilp_para.lambda[n][m] = alpha / beta;
      ilp_para.delta2inv[n][m] = 1.0 / (delta * delta);
      ilp_para.S[n][m] = S;
      ilp_para.rcutsq_ilp[n][m] = rcut_ilp * rcut_ilp;
      ilp_para.rcut_global[n][m] = rcut_global;
      float meV = 1e-3 * S;
      ilp_para.C[n][m] *= meV;
      ilp_para.C_6[n][m] *= meV;
      ilp_para.epsilon[n][m] *= meV;

      if (rc < rcut_global)
        rc = rcut_global;
    }
  }

  // Optional footer: zbl N; then N lines "element_i element_j inner outer".
  // EOF preserves the exact legacy IWP path. Each unordered pair occurs once.
  char footer[32];
  const bool has_zbl_footer = fscanf(fid_ilp, "%31s", footer) != EOF;
  if (require_zbl && !has_zbl_footer) {
    PRINT_INPUT_ERROR("nep_iwp_zbl requires an explicit zbl pair configuration.\n");
  }
  if (has_zbl_footer) {
    if (std::strcmp(footer, "zbl") != 0) {
      PRINT_INPUT_ERROR("Unknown IWP footer; expected zbl N.\n");
    }
    int number_of_pairs = 0;
    PRINT_SCANF_ERROR(fscanf(fid_ilp, "%d", &number_of_pairs), 1,
      "Reading IWP ZBL pair count.");
    if (number_of_pairs < 1 || number_of_pairs > num_types*(num_types+1)/2) {
      PRINT_INPUT_ERROR("Invalid number of IWP ZBL unordered pairs.\n");
    }
    for (int pair = 0; pair < number_of_pairs; ++pair) {
      char first[10], second[10];
      double inner, outer;
      PRINT_SCANF_ERROR(fscanf(fid_ilp, "%9s%9s%lf%lf", first, second, &inner, &outer), 4,
        "Reading IWP ZBL pair: element_i element_j inner outer.");
      int i = -1, j = -1;
      for (int n = 0; n < num_types; ++n) {
        if (ilp_elements[n] == first) i = n;
        if (ilp_elements[n] == second) j = n;
      }
      if (i < 0 || j < 0) PRINT_INPUT_ERROR("Unknown element in IWP ZBL pair.\n");
      if (!std::isfinite(inner) || !std::isfinite(outer) ||
          !(inner > 0 && outer > inner) ||
          outer > ilp_para.rcut_global[i][j] || outer > ilp_para.rcut_global[j][i] ||
          !(float(inner) > 0 && float(outer) > float(inner))) {
        PRINT_INPUT_ERROR("IWP ZBL requires finite 0 < inner < outer <= both IWP pair cutoffs.\n");
      }
      if (ilp_para.zbl_outer[i][j] > 0) {
        PRINT_INPUT_ERROR("Duplicate unordered IWP ZBL pair.\n");
      }
      const int zi = iwp_m6x6::atomic_number(first);
      const int zj = iwp_m6x6::atomic_number(second);
      const float zizj = float(14.3996454784255*zi*zj);
      const float a_inv = float((std::pow(double(zi),0.23)+std::pow(double(zj),0.23))/0.46850);
      ilp_para.zbl_inner[i][j] = ilp_para.zbl_inner[j][i] = float(inner);
      ilp_para.zbl_outer[i][j] = ilp_para.zbl_outer[j][i] = float(outer);
      ilp_para.zbl_zizj[i][j] = ilp_para.zbl_zizj[j][i] = zizj;
      ilp_para.zbl_a_inv[i][j] = ilp_para.zbl_a_inv[j][i] = a_inv;
      printf("IWP ZBL %s-%s: pure below %g A, original IWP above %g A (C2 blend).\n",
        first, second, inner, outer);
    }
    if (fscanf(fid_ilp, "%31s", footer) != EOF) {
      PRINT_INPUT_ERROR("Unexpected data after IWP ZBL footer.\n");
    }
  }

  // read NEP group method from nep map file
  PRINT_SCANF_ERROR(fscanf(fid_nep_map, "%d", &nep_group_method), 1, 
  "Reading error for NEP group method.");
  printf("Use group method %d to identify molecule for NEP.\n", nep_group_method);

  // read the number of NEP file
  PRINT_SCANF_ERROR(fscanf(fid_nep_map, "%d", &num_nep), 1, 
  "Reading error for the number of NEP file.");
  printf("NEP file number: %d\n", num_nep);

  // init parameter vectors
  parambs.resize(num_nep);
  annmbs.resize(num_nep);
  max_nep_rc = 0.0;

  // init type map cpu
  type_map_cpu.resize(num_types * num_nep, -1);
  
  // read NEP parameter from each NEP file
  std::vector<std::vector<float>> all_ann_para({});     // save paras
  for (int i = 0; i < num_nep; ++i) {
    printf("\nReading NEP %d.\n", i);
    char nep_file[100];
    int count = fscanf(fid_nep_map, "%s", nep_file);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP filename");

    std::ifstream input(nep_file);
    if (!input.is_open()) {
      std::cout << "Failed to open " << nep_file << std::endl;
      exit(1);
    }

    // nep4 1 C
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() < 3) {
      std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
      exit(1);
    }
    if (tokens[0] != "nep4") {
      std::cout << tokens[0]
                << " is an unsupported NEP model. We only support NEP4 models now."
                << std::endl;
      exit(1);
    }
    parambs[i].num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
    if (tokens.size() != 2 + parambs[i].num_types) {
      std::cout << "The first line of nep.txt should have " << parambs[i].num_types << " atom symbols."
                << std::endl;
      exit(1);
    }

    if (parambs[i].num_types == 1) {
      printf("Use the NEP%d potential with %d atom type.\n", parambs[i].version, parambs[i].num_types);
    } else {
      printf("Use the NEP%d potential with %d atom types.\n", parambs[i].version, parambs[i].num_types);
    }

    for (int n = 0; n < parambs[i].num_types; ++n) {
      int atomic_number = 0;
      for (int m = 0; m < NUM_ELEMENTS; ++m) {
        if (tokens[2 + n] == ELEMENTS[m]) {
          atomic_number = m + 1;
          break;
        }
      }
      parambs[i].atomic_numbers[n] = atomic_number - 1;
      printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), atomic_number);

      // update type map
      // for example: if ilp elements is C N B, element in nep 0 is C, elements in nep 1 are B N
      // type map should be [0,    -1,    -1,    -1,    1,    0]
      //   ilp element       C      N      B      C     N     B
      //   nep 0 element     C(0)   null   null
      //   nep 1 element                          null  N(1)  B(0)
      for (int m = 0; m < num_types; ++m) {
        if (tokens[2 + n] == ilp_elements[m]) {
          type_map_cpu[m + i * num_types] = n;
        }
      }
    }

    // cutoff 4.2 3.7 80 47 1
    tokens = get_tokens(input);
    if (tokens.size() != 5 && tokens.size() != 8) {
      std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                   "[radial_factor] [angular_factor] [zbl_factor].\n";
      exit(1);
    }
    parambs[i].rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
    parambs[i].rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
    printf("    radial cutoff = %g A.\n", parambs[i].rc_radial);
    printf("    angular cutoff = %g A.\n", parambs[i].rc_angular);
    // save the max rc_radial, same to max rc
    max_nep_rc = max(max_nep_rc, parambs[i].rc_radial);

    int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
    printf("    MN_radial = %d.\n", MN_radial);
    if (MN_radial > 819) {
      std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
                << std::endl;
      exit(1);
    }
    parambs[i].MN_radial = int(ceil(MN_radial * 1.25));
    parambs[i].MN_angular = int(ceil(MN_angular * 1.25));
    max_MN_radial = max(max_MN_radial, parambs[i].MN_radial);
    max_MN_angular = max(max_MN_angular, parambs[i].MN_angular);
    printf("    enlarged MN_radial = %d.\n", parambs[i].MN_radial);
    printf("    enlarged MN_angular = %d.\n", parambs[i].MN_angular);

    if (tokens.size() == 8) {
      parambs[i].typewise_cutoff_radial_factor = get_double_from_token(tokens[5], __FILE__, __LINE__);
      parambs[i].typewise_cutoff_angular_factor = get_double_from_token(tokens[6], __FILE__, __LINE__);
      if (parambs[i].typewise_cutoff_radial_factor > 0.0f) {
        parambs[i].use_typewise_cutoff = true;
      }
    }

    // n_max 10 8
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
      exit(1);
    }
    parambs[i].n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
    parambs[i].n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
    max_n_max_angular = max(max_n_max_angular, parambs[i].n_max_angular);
    printf("    n_max_radial = %d.\n", parambs[i].n_max_radial);
    printf("    n_max_angular = %d.\n", parambs[i].n_max_angular);

    // basis_size 10 8
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
                << std::endl;
      exit(1);
    }
    parambs[i].basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
    parambs[i].basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
    printf("    basis_size_radial = %d.\n", parambs[i].basis_size_radial);
    printf("    basis_size_angular = %d.\n", parambs[i].basis_size_angular);

    // l_max
    tokens = get_tokens(input);
    if (tokens.size() != 4) {
      std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
      exit(1);
    }

    parambs[i].L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
    printf("    l_max_3body = %d.\n", parambs[i].L_max);
    parambs[i].num_L = parambs[i].L_max;

    int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
    int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
    printf("    l_max_4body = %d.\n", L_max_4body);
    printf("    l_max_5body = %d.\n", L_max_5body);
    if (L_max_4body == 2) {
      parambs[i].num_L += 1;
    }
    if (L_max_5body == 1) {
      parambs[i].num_L += 1;
    }

    parambs[i].dim_angular = (parambs[i].n_max_angular + 1) * parambs[i].num_L;

    // ANN
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be ANN num_neurons 0." << std::endl;
      exit(1);
    }
    annmbs[i].num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
    annmbs[i].dim = (parambs[i].n_max_radial + 1) + parambs[i].dim_angular;
    nep_model_type = parambs[i].model_type;
    max_dim = max(max_dim, annmbs[i].dim);
    printf("    ANN = %d-%d-1.\n", annmbs[i].dim, annmbs[i].num_neurons1);

    // calculated parameters:
    parambs[i].rcinv_radial = 1.0f / parambs[i].rc_radial;
    parambs[i].rcinv_angular = 1.0f / parambs[i].rc_angular;
    parambs[i].num_types_sq = parambs[i].num_types * parambs[i].num_types;

    annmbs[i].num_para_ann =
      (annmbs[i].dim + 2) * annmbs[i].num_neurons1 * parambs[i].num_types + 1;
    printf("    number of neural network parameters = %d.\n", annmbs[i].num_para_ann);
    int num_para_descriptor =
      parambs[i].num_types_sq * ((parambs[i].n_max_radial + 1) * (parambs[i].basis_size_radial + 1) +
                             (parambs[i].n_max_angular + 1) * (parambs[i].basis_size_angular + 1));
    printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
    annmbs[i].num_para = annmbs[i].num_para_ann + num_para_descriptor;
    printf("    total number of parameters = %d.\n", annmbs[i].num_para);

    parambs[i].num_c_radial =
      parambs[i].num_types_sq * (parambs[i].n_max_radial + 1) * (parambs[i].basis_size_radial + 1);

    // NN and descriptor parameters
    std::vector<float> parameters(annmbs[i].num_para);
    for (int n = 0; n < annmbs[i].num_para; ++n) {
      tokens = get_tokens(input);
      parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    std::vector<float> descriptor_parameters = get_descriptor_parameters_type_pair(
      parameters,
      annmbs[i].num_para_ann,
      parambs[i].num_types,
      parambs[i].n_max_radial,
      parambs[i].n_max_angular,
      parambs[i].basis_size_radial,
      parambs[i].basis_size_angular);
    for (int n = 0; n < num_para_descriptor; ++n) {
      parameters[annmbs[i].num_para_ann + n] = descriptor_parameters[n];
    }
    all_ann_para.push_back(parameters);
    for (int d = 0; d < annmbs[i].dim; ++d) {
      tokens = get_tokens(input);
      parambs[i].q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }

  }

  // check ilp rc and nep rc, make sure ilp's larger than nep's
  if (max_nep_rc > rc) {
    printf("ERROR: Please make sure cutoff of ILP is larger than it of NEP.\n");
    exit(1);
  }

  // read nep map to identify the nep for each group
  int num_nep_group = 0;
  PRINT_SCANF_ERROR(fscanf(fid_nep_map, "%d", &num_nep_group), 1, 
  "Reading error for the number of nep group.");
  nep_map_cpu.resize(num_nep_group);
  for (int i = 0; i < num_nep_group; ++i) {
    int nep_i = 0;    // which nep this group use
    int count = fscanf(fid_nep_map, "%d", &nep_i);
    PRINT_SCANF_ERROR(count, 1, "reading error for nep number of group.");
    if (nep_i >= num_nep) {
      if (num_nep == 1) {
        printf("There is only 1 nep file, but you set group %d of group method %d \
        to nep %d", i, nep_group_method, nep_i);
      } else {
        printf("There are %d nep files, but you set group %d of group method %d \
        to nep %d", num_nep, i, nep_group_method, nep_i);
      }
      exit(1);
    }
    nep_map_cpu[i] = nep_i;
    printf("group %d of group method %d uses NEP %d.\n", i, nep_group_method, nep_i);
  }
  // cp two maps to gpu
  nep_map.resize(num_nep_group);
  type_map.resize(num_types * num_nep);
  nep_map.copy_from_host(nep_map_cpu.data());
  type_map.copy_from_host(type_map_cpu.data());


  // initialize ilp neighbor lists and some temp vectors
  int max_neighbor_number = min(num_atoms, IWP_M6X6_CUDA_MAX_NL_ILP_NEP_TMD);
  ilp_data.NN.resize(num_atoms);
  ilp_data.NL.resize(num_atoms * max_neighbor_number);
  ilp_data.cell_count.resize(num_atoms);
  ilp_data.cell_count_sum.resize(num_atoms);
  ilp_data.cell_contents.resize(num_atoms);

  // init ilp neighbor list
  ilp_data.ilp_NN.resize(num_atoms);
  ilp_data.ilp_NL.resize(num_atoms * IWP_M6X6_MAX_ILP_NEIGHBOR_TMD);
  ilp_data.reduce_NL.resize(num_atoms * max_neighbor_number);

  ilp_data.f12x.resize(num_atoms * max_neighbor_number);
  ilp_data.f12y.resize(num_atoms * max_neighbor_number);
  ilp_data.f12z.resize(num_atoms * max_neighbor_number);

  ilp_data.f12x_ilp_neigh.resize(num_atoms * IWP_M6X6_MAX_ILP_NEIGHBOR_TMD);
  ilp_data.f12y_ilp_neigh.resize(num_atoms * IWP_M6X6_MAX_ILP_NEIGHBOR_TMD);
  ilp_data.f12z_ilp_neigh.resize(num_atoms * IWP_M6X6_MAX_ILP_NEIGHBOR_TMD);

  // init constant cutoff coeff
  float h_tap_coeff[8] = \
    {1.0f, 0.0f, 0.0f, 0.0f, -35.0f, 84.0f, -70.0f, 20.0f};
  CHECK(gpuMemcpyToSymbol(IWP_M6X6_Tap_coeff, h_tap_coeff, 8 * sizeof(float)));

  // set ilp_flag to 1
  ilp_flag = 1;

  // initialize nep neighbor lists
  nep_data.f12x.resize(num_atoms * max_MN_angular);
  nep_data.f12y.resize(num_atoms * max_MN_angular);
  nep_data.f12z.resize(num_atoms * max_MN_angular);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * max_MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * max_MN_angular);
  nep_data.Fp.resize(num_atoms * max_dim);
  nep_data.sum_fxyz.resize(num_atoms * (max_n_max_angular + 1) * NUM_OF_ABC);
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);

  // updata nep parameters and ann parameters
  // calc buffer size
  int parambs_size = num_nep * (IWP_M6X6_H_PAR_OFFSET) * IWP_M6X6_SIZEOF_INT;
  int annmbs_size = num_nep * (IWP_M6X6_H_ANN_OFFSET) * IWP_M6X6_SIZEOF_INT;
  for (int i = 0; i < num_nep; ++i) {
    // add q_scaler and atomic_number buffers
    parambs_size += (annmbs[i].dim + parambs[i].num_types) * IWP_M6X6_SIZEOF_INT;
    // add w0, b0, w1, c buffer
    annmbs_size += (annmbs[i].num_para - 1) * IWP_M6X6_SIZEOF_INT;   // rm b1
  }
  int all_para_size = parambs_size + annmbs_size;

  // init buffers in cpu and gpu
  char* para_buffer_cpu = (char*) malloc(all_para_size);
  nep_data.para_buffer_gpu.resize(all_para_size);
  char* para_buffer_gpu = nep_data.para_buffer_gpu.data();

  // save head pointers
  h_parambs = para_buffer_gpu;
  h_annmbs = h_parambs + num_nep * IWP_M6X6_H_PAR_OFFSET * IWP_M6X6_SIZEOF_INT;

  // set parameters data in cpu buffer
  int* para_buf_w = (int*) para_buffer_cpu;     // a write ptr to cpu buffer
  int* para_buf_ptrw = para_buf_w;              // a write ptr to write the ptrs of para

  // parambs head
  for (int i = 0; i < num_nep; ++i) {
    int use_typewise_cutoff_int = parambs[i].use_typewise_cutoff;
    memcpy(para_buf_w + IWP_M6X6_UTC    , &use_typewise_cutoff_int, IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_TCRF   , &(parambs[i].typewise_cutoff_radial_factor), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_TCAF   , &(parambs[i].typewise_cutoff_angular_factor), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_VERSION, &(parambs[i].version), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_RCR    , &(parambs[i].rc_radial), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_RCA    , &(parambs[i].rc_angular), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_RCIR   , &(parambs[i].rcinv_radial), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_RCIA   , &(parambs[i].rcinv_angular), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_MNR    , &(parambs[i].MN_radial), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_MNA    , &(parambs[i].MN_angular), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_NMAXR  , &(parambs[i].n_max_radial), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_NMAXA  , &(parambs[i].n_max_angular), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_LMAX   , &(parambs[i].L_max), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_DIMA   , &(parambs[i].dim_angular), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_NUML   , &(parambs[i].num_L), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_BSR    , &(parambs[i].basis_size_radial), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_BSA    , &(parambs[i].basis_size_angular), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_NTS    , &(parambs[i].num_types_sq), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_NCR    , &(parambs[i].num_c_radial), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_NT     , &(parambs[i].num_types), IWP_M6X6_SIZEOF_INT);
    para_buf_w += IWP_M6X6_H_PAR_OFFSET;    // skip 2 pointers: IWP_M6X6_PTRQS IWP_M6X6_PTRAN
  }

  // annmbs head
  for (int i = 0; i < num_nep; ++i) {
    memcpy(para_buf_w + IWP_M6X6_ANNDIM , &(annmbs[i].dim), IWP_M6X6_SIZEOF_INT);
    memcpy(para_buf_w + IWP_M6X6_NNEUR  , &(annmbs[i].num_neurons1), IWP_M6X6_SIZEOF_INT);
    int b1_pos =
      (annmbs[i].dim + 2) * annmbs[i].num_neurons1 * parambs[i].num_types;
    memcpy(para_buf_w + IWP_M6X6_OUTB1  , &(all_ann_para[i][b1_pos]), IWP_M6X6_SIZEOF_INT);
    para_buf_w += IWP_M6X6_H_ANN_OFFSET;  // skip 4 pointers: IWP_M6X6_PTRC IWP_M6X6_PTRW0 IWP_M6X6_PTRB0 IWP_M6X6_PTRW1 and an empty
  }
  
  // move gpu buffer pointer
  para_buffer_gpu += num_nep * (IWP_M6X6_H_PAR_OFFSET + IWP_M6X6_H_ANN_OFFSET) * IWP_M6X6_SIZEOF_INT;

  // q_scaler
  for (int i = 0; i < num_nep; ++i) {
    // update pointer q_scaler
    memcpy(para_buf_ptrw + i * IWP_M6X6_H_PAR_OFFSET + IWP_M6X6_PTRQS, &para_buffer_gpu, IWP_M6X6_SIZEOF_POINTER);

    int qs_offset = annmbs[i].dim;
    memcpy(para_buf_w, parambs[i].q_scaler, qs_offset * IWP_M6X6_SIZEOF_INT);
    para_buf_w += qs_offset;
    para_buffer_gpu += qs_offset * IWP_M6X6_SIZEOF_INT;
  }

  // atomic_numbers
  for (int i = 0; i < num_nep; ++i) {
    // update pointer atomic number
    memcpy(para_buf_ptrw + i * IWP_M6X6_H_PAR_OFFSET + IWP_M6X6_PTRAN, &para_buffer_gpu, IWP_M6X6_SIZEOF_POINTER);

    int an_offset = parambs[i].num_types;
    memcpy(para_buf_w, parambs[i].atomic_numbers, an_offset * IWP_M6X6_SIZEOF_INT);
    para_buf_w += an_offset;
    para_buffer_gpu += an_offset * IWP_M6X6_SIZEOF_INT;
  }

  // move cpu para buffer ptr w to the annmb head
  para_buf_ptrw += num_nep * IWP_M6X6_H_PAR_OFFSET;

  // w0
  for (int i = 0; i < num_nep; ++i) {
    // update pointer w0
    memcpy(para_buf_ptrw + i * IWP_M6X6_H_ANN_OFFSET + IWP_M6X6_PTRW0, &para_buffer_gpu, IWP_M6X6_SIZEOF_POINTER);

    int w0_offset = annmbs[i].num_neurons1 * annmbs[i].dim;
    int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1;
    for (int t = 0; t < parambs[i].num_types; ++t) {
      memcpy(para_buf_w, &(all_ann_para[i][t * t_offset]), w0_offset * IWP_M6X6_SIZEOF_INT);
      para_buf_w += w0_offset;
    }
    para_buffer_gpu += parambs[i].num_types * w0_offset * IWP_M6X6_SIZEOF_INT;

  }

  // b0
  for (int i = 0; i < num_nep; ++i) {
    // update pointer b0
    memcpy(para_buf_ptrw + i * IWP_M6X6_H_ANN_OFFSET + IWP_M6X6_PTRB0, &para_buffer_gpu, IWP_M6X6_SIZEOF_POINTER);

    int b0_offset = annmbs[i].num_neurons1;
    int b0_base = annmbs[i].num_neurons1 * annmbs[i].dim;
    int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1;
    for (int t = 0; t < parambs[i].num_types; ++t) {
      memcpy(para_buf_w, &(all_ann_para[i][b0_base + t * t_offset]), b0_offset * IWP_M6X6_SIZEOF_INT);
      para_buf_w += b0_offset;
    }
    para_buffer_gpu += parambs[i].num_types * b0_offset * IWP_M6X6_SIZEOF_INT;

  }

  // w1
  for (int i = 0; i < num_nep; ++i) {
    // update pointer w1
    memcpy(para_buf_ptrw + i * IWP_M6X6_H_ANN_OFFSET + IWP_M6X6_PTRW1, &para_buffer_gpu, IWP_M6X6_SIZEOF_POINTER);

    int w1_offset = annmbs[i].num_neurons1;
    int w1_base = annmbs[i].num_neurons1 * (annmbs[i].dim + 1);
    int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1;
    for (int t = 0; t < parambs[i].num_types; ++t) {
      memcpy(para_buf_w, &(all_ann_para[i][w1_base + t * t_offset]), w1_offset * IWP_M6X6_SIZEOF_INT);
      para_buf_w += w1_offset;
    }
    para_buffer_gpu += parambs[i].num_types * w1_offset * IWP_M6X6_SIZEOF_INT;

  }

  // c
  for (int i = 0; i < num_nep; ++i) {
    // update pointer c
    memcpy(para_buf_ptrw + i * IWP_M6X6_H_ANN_OFFSET + IWP_M6X6_PTRC, &para_buffer_gpu, IWP_M6X6_SIZEOF_POINTER);

    int c_offset = annmbs[i].num_para - annmbs[i].num_para_ann;
    int c_base = annmbs[i].num_para_ann;
    memcpy(para_buf_w, &(all_ann_para[i][c_base]), c_offset * IWP_M6X6_SIZEOF_INT);
    para_buf_w += c_offset;
    para_buffer_gpu += c_offset * IWP_M6X6_SIZEOF_INT;
  }


  // cp parameters from cpu to gpu
  nep_data.para_buffer_gpu.copy_from_host(para_buffer_cpu, all_para_size);

  // free cpu buffer and set ptrs to null
  free(para_buffer_cpu);
  para_buffer_cpu = nullptr;
  para_buffer_gpu = nullptr;
  para_buf_w= nullptr;
  para_buf_ptrw = nullptr;

}

IWP_NEP_M6X6::~IWP_NEP_M6X6(void)
{
  // nothing
}

static __device__ __forceinline__ float calc_Tap(const float r_ij, const float Rcutinv)
{
  float Tap, r;

  r = r_ij * Rcutinv;
  if (r >= 1.0f) {
    Tap = 0.0f;
  } else {
    Tap = IWP_M6X6_Tap_coeff[7];
    for (int i = 6; i >= 0; --i) {
      Tap = Tap * r + IWP_M6X6_Tap_coeff[i];
    }
  }

  return Tap;
}

// calculate the derivatives of long-range cutoff term
static __device__ __forceinline__ float calc_dTap(const float r_ij, const float Rcutinv)
{
  float dTap, r;
  
  r = r_ij * Rcutinv;
  if (r >= 1.0f) {
    dTap = 0.0f;
  } else {
    dTap = 7.0f * IWP_M6X6_Tap_coeff[7];
    for (int i = 6; i > 0; --i) {
      dTap = dTap * r + i * IWP_M6X6_Tap_coeff[i];
    }
    dTap *= Rcutinv;
  }

  return dTap;
}

static __device__ __forceinline__ void iwp_device_trap()
{
#ifdef USE_HIP
  __builtin_trap();
#else
  asm("trap;");
#endif
}

// IWP pair neighbors belong to different wires; NEP and normals stay within a wire.
static __global__ void gpu_find_neighbor_ON1_ilp_nep(
  const int* nep_map,
  const int* type_map,
  const int* group_label_nep,
  void* h_parambs,
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int* __restrict__ type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  int* NN_nep_radial,
  int* NL_nep_radial,
  int* NN_nep_angular,
  int* NL_nep_angular,
  const int* group_label_ilp,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const float ilp_cutoff_square,
  const IWP_M6X6_Para ilp_para,
  const float* iwp_normal_sign,
  int* normal_number, int* normal_list,
  const int radial_capacity, const int angular_capacity)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[group_label_nep[n1]];
    float* paramb = (float*)h_parambs + nep_id * IWP_M6X6_H_PAR_OFFSET;
    float rc_radial = paramb[IWP_M6X6_RCR];
    float rc_angular = paramb[IWP_M6X6_RCA];
    float rc_radial_sq = rc_radial * rc_radial;
    float rc_angular_sq = rc_angular * rc_angular;
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int ilp_count_diff = 0;   // ilp neighbor in different layer to calc energy
    int nep_count_radial = 0;
    int nep_count_angular = 0;
    float best_distance_sq[4] = {1e30f, 1e30f, 1e30f, 1e30f};
    int best_index[4] = {-1, -1, -1, -1};
    const int type1 = type[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    int z_begin, z_end;
    int y_begin, y_end;
    int x_begin, x_end;
    get_neighbor_cell_offset_range(box.pbc_z, nz, z_begin, z_end);
    get_neighbor_cell_offset_range(box.pbc_y, ny, y_begin, y_end);
    get_neighbor_cell_offset_range(box.pbc_x, nx, x_begin, x_end);

    // get radial descriptors
    for (int k = z_begin; k < z_end; ++k) {
      for (int j = y_begin; j < y_end; ++j) {
        for (int i = x_begin; i < x_end; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            // neighbors in different layers
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              float d2 = (float)x12 * (float)x12 + (float)y12 * (float)y12 + (float)z12 * (float)z12;

              const bool different_layer = group_label_ilp[n1] != group_label_ilp[n2];
              const int type2 = type[n2];
              if (!different_layer && iwp_normal_sign[type1] * iwp_normal_sign[type2] < 0.0f &&
                  d2 > 1e-20f && d2 < ilp_para.rcutsq_ilp[type1][type2] && d2 < best_distance_sq[3]) {
                int insertion = 3;
                while (insertion > 0 && d2 < best_distance_sq[insertion - 1]) {
                  best_distance_sq[insertion] = best_distance_sq[insertion - 1];
                  best_index[insertion] = best_index[insertion - 1];
                  --insertion;
                }
                best_distance_sq[insertion] = d2;
                best_index[insertion] = n2;
              }
              if (different_layer && d2 < ilp_cutoff_square) {
                if (ilp_count_diff >= IWP_M6X6_CUDA_MAX_NL_ILP_NEP_TMD) {
                  printf("ERROR IWP interwire neighbor capacity exceeded.\n");
                  iwp_device_trap();
                }
                NL[ilp_count_diff++ * N + n1] = n2;
              } else if (!different_layer) {
                if (d2 < rc_radial_sq) {
                  if (nep_count_radial >= radial_capacity) { printf("ERROR IWP NEP radial capacity exceeded.\n"); iwp_device_trap(); }
                  NL_nep_radial[nep_count_radial++ * N + n1] = n2;
                }
                if (d2 < rc_angular_sq) {
                  if (nep_count_angular >= angular_capacity) { printf("ERROR IWP NEP angular capacity exceeded.\n"); iwp_device_trap(); }
                  NL_nep_angular[nep_count_angular++ * N + n1] = n2;
                }
              }

            }
          }
        }
      }
    }
    int normal_count = 0;
    for (int k = 0; k < 4; ++k) {
      if (best_index[k] >= 0) normal_list[normal_count++ * N + n1] = best_index[k];
    }
    normal_number[n1] = normal_count;
    if (normal_count != 4) {
      printf("ERROR IWP normal: atom %d has %d same-wire unlike neighbors within rcut_normal; four are required.\n", n1, normal_count);
    }
    NN[n1] = ilp_count_diff;
    NN_nep_radial[n1] = nep_count_radial;
    NN_nep_angular[n1] = nep_count_angular;
  }
}

void iwp_m6x6_find_neighbor_ilp_nep(
  const int N1,
  const int N2,
  const int* nep_map,
  const int* type_map,
  const int* group_label_nep,
  void* h_parambs,
  float rc,
  Box& box,
  const int* group_label_ilp,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL,
  GPU_Vector<int>& NN_nep_radial,
  GPU_Vector<int>& NL_nep_radial,
  GPU_Vector<int>& NN_nep_angular,
  GPU_Vector<int>& NL_nep_angular,
  const IWP_M6X6_Para ilp_para,
  const float* normal_sign,
  GPU_Vector<int>& normal_number, GPU_Vector<int>& normal_list,
  const double normal_cutoff, const double nep_cutoff)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double search_cutoff = std::max(double(rc), std::max(normal_cutoff, nep_cutoff));
  const double rc_cell_list = 0.5 * search_cutoff;
  const double rc_inv_cell_list = 2.0 / search_cutoff;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);
  for (int d = 0; d < 3; ++d) num_bins[d] = std::max(1, num_bins[d]);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1_ilp_nep<<<grid_size, block_size>>>(
    nep_map,
    type_map,
    group_label_nep,
    h_parambs,
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    NN_nep_radial.data(),
    NL_nep_radial.data(),
    NN_nep_angular.data(),
    NL_nep_angular.data(),
    group_label_ilp,
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc, ilp_para, normal_sign, normal_number.data(), normal_list.data(),
    NL_nep_radial.size() / N, NL_nep_angular.size() / N);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list<<<N, min(1024, MN), MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
}

// calculate the van der Waals force and energy
static __device__ void calc_vdW(
  float r,
  float rinv,
  float rsq,
  float d,
  float d_Seff,
  float C_6,
  float Tap,
  float dTap,
  float &p2_vdW,
  float &f2_vdW)
{
  float r2inv, r6inv, r8inv;
  float Vilp;
  double TSvdw_double, TSvdwinv_double;
  float TSvdwinv_float;
  float fpair, fsum;

  r2inv = 1.0f / rsq;
  r6inv = r2inv * r2inv * r2inv;
  r8inv = r2inv * r6inv;

  // TSvdw = 1.0 + exp(-d_Seff * r + d);
  // TSvdw = 1.0f + expf(-d_Seff * r + d);
  TSvdw_double = 1.0 + exp((double) (-d_Seff * r + d));
  // TSvdwinv = 1.0f / TSvdw;
  TSvdwinv_double = 1.0 / TSvdw_double;
  TSvdwinv_float = (float) TSvdwinv_double;
  Vilp = -C_6 * r6inv * TSvdwinv_float;

  // derivatives
  fpair = (-6.0f + d_Seff * (1.0f - TSvdwinv_float) * r ) * C_6 * TSvdwinv_float * r8inv;
  fsum = fpair * Tap - Vilp * dTap * rinv;

  p2_vdW = Tap * Vilp;
  f2_vdW = fsum;
}

// force evaluation kernel
static __global__ void gpu_find_force(
  IWP_M6X6_Para ilp_para,
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int *g_neighbor_number,
  const int *g_neighbor_list,
  int *g_ilp_neighbor_number,
  int *g_ilp_neighbor_list,
  const int *group_label,
  const int *g_type,
  const double *__restrict__ g_x,
  const double *__restrict__ g_y,
  const double *__restrict__ g_z,
  double *g_fx,
  double *g_fy,
  double *g_fz,
  double *g_virial,
  double *g_potential,
  float *g_f12x,
  float *g_f12y,
  float *g_f12z,
  float *g_f12x_ilp_neigh,
  float *g_f12y_ilp_neigh,
  float *g_f12z_ilp_neigh,
  const float* iwp_normal_sign)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  float s_fx = 0.0f;                                   // force_x
  float s_fy = 0.0f;                                   // force_y
  float s_fz = 0.0f;                                   // force_z
  float s_pe = 0.0f;                                   // potential energy
  float s_sxx = 0.0f;                                  // virial_stress_xx
  float s_sxy = 0.0f;                                  // virial_stress_xy
  float s_sxz = 0.0f;                                  // virial_stress_xz
  float s_syx = 0.0f;                                  // virial_stress_yx
  float s_syy = 0.0f;                                  // virial_stress_yy
  float s_syz = 0.0f;                                  // virial_stress_yz
  float s_szx = 0.0f;                                  // virial_stress_zx
  float s_szy = 0.0f;                                  // virial_stress_zy
  float s_szz = 0.0f;                                  // virial_stress_zz

  float r = 0.0f;
  float rsq = 0.0f;
  float Rcut = 0.0f;

  if (n1 < N2) {
    double x12d, y12d, z12d;
    float x12f, y12f, z12f;
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    float delkix_half[IWP_M6X6_MAX_ILP_NEIGHBOR_TMD] = {0.0f};
    float delkiy_half[IWP_M6X6_MAX_ILP_NEIGHBOR_TMD] = {0.0f};
    float delkiz_half[IWP_M6X6_MAX_ILP_NEIGHBOR_TMD] = {0.0f};

    // calculate the normal
    int cont = 0;
    float normal[3];
    float dnormdri[3][3];
    float dnormal[3][IWP_M6X6_MAX_ILP_NEIGHBOR_TMD][3];

    float vet[IWP_M6X6_MAX_ILP_NEIGHBOR_TMD][3];
    int id, ip, m;
    for (id = 0; id < 3; ++id) {
      normal[id] = 0.0f;
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[ip][id] = 0.0f;
        for (m = 0; m < IWP_M6X6_MAX_ILP_NEIGHBOR_TMD; ++m) {
          dnormal[id][m][ip] = 0.0f;
          vet[m][id] = 0.0f;
        }
      }
    }

    int ilp_neighbor_number = g_ilp_neighbor_number[n1];
    for (int i1 = 0; i1 < ilp_neighbor_number; ++i1) {
      int n2_ilp = g_ilp_neighbor_list[n1 + number_of_particles * i1];
      x12d = g_x[n2_ilp] - x1;
      y12d = g_y[n2_ilp] - y1;
      z12d = g_z[n2_ilp] - z1;
      apply_mic(box, x12d, y12d, z12d);
      vet[cont][0] = float(x12d);
      vet[cont][1] = float(y12d);
      vet[cont][2] = float(z12d);
      ++cont;

      delkix_half[i1] = float(x12d) * 0.5f;
      delkiy_half[i1] = float(y12d) * 0.5f;
      delkiz_half[i1] = float(z12d) * 0.5f;
    }
    
    if (!iwp_m6x6::calc_radial_normal_and_derivatives<IWP_M6X6_MAX_ILP_NEIGHBOR_TMD>(
          vet, cont, iwp_normal_sign[type1], normal, dnormdri, dnormal)) {
      printf("ERROR IWP normal: failed to evaluate atom %d from %d neighbors.\n", n1, cont);
      return;
    }


    // calculate energy and force
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_neighbor_list[index];
      int type2 = g_type[n2];

      x12d = g_x[n2] - x1;
      y12d = g_y[n2] - y1;
      z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);

      // save x12, y12, z12 in float
      x12f = float(x12d);
      y12f = float(y12d);
      z12f = float(z12d);

      // calculate distance between atoms
      rsq = x12f * x12f + y12f * y12f + z12f * z12f;
      if (rsq <= 0.0f) {
        printf("ERROR IWP: exactly overlapping interwire atoms %d and %d.\n", n1, n2);
        iwp_device_trap();
      }
      r = sqrtf(rsq);
      Rcut = ilp_para.rcut_global[type1][type2];

      if (r >= Rcut) {
        continue;
      }

      // calc att
      float Tap, dTap, rinv;
      float Rcutinv = 1.0f / Rcut;
      rinv = 1.0f / r;
      Tap = calc_Tap(r, Rcutinv);
      dTap = calc_dTap(r, Rcutinv);

      float blend = 1.0f, blend_p = 0.0f;
      iwp_m6x6::zbl_blend_weight(r, ilp_para.zbl_inner[type1][type2],
        ilp_para.zbl_outer[type1][type2], blend, blend_p);

      float p2_vdW = 0.0f, f2_vdW = 0.0f;
      if (blend > 0.0f) calc_vdW(
        r,
        rinv,
        rsq,
        ilp_para.d[type1][type2],
        ilp_para.d_Seff[type1][type2],
        ilp_para.C_6[type1][type2],
        Tap,
        dTap,
        p2_vdW,
        f2_vdW);

      if (blend < 1.0f || blend_p != 0.0f) {
        float zbl_energy, zbl_derivative;
        iwp_m6x6::zbl_core(r, ilp_para.zbl_zizj[type1][type2],
          ilp_para.zbl_a_inv[type1][type2], zbl_energy, zbl_derivative);
        // f2 = -dV/dr / r. The repulsive directed part adds its w' term below.
        f2_vdW = blend*f2_vdW -
          ((1.0f-blend)*zbl_derivative + blend_p*(p2_vdW-zbl_energy))*rinv;
        p2_vdW = blend*p2_vdW + (1.0f-blend)*zbl_energy;
      }
      
      float f12x = -f2_vdW * x12f * 0.5f;
      float f12y = -f2_vdW * y12f * 0.5f;
      float f12z = -f2_vdW * z12f * 0.5f;
      float f21x = -f12x;
      float f21y = -f12y;
      float f21z = -f12z;

      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      s_pe += p2_vdW * 0.5f;
      
      // if (n1 == 0)
      // printf("n2[%d] dx[%.16f] dy[%.16f] dz[%.16f] r[%.16f] rsq[%.16f] rinv[%.16f] tap[%.16f] att[%.16f]\n", 
      // n2, x12d, y12d, z12d, r, rsq, rinv, Tap, p2_vdW);
      s_sxx += x12f * f21x;
      s_sxy += x12f * f21y;
      s_sxz += x12f * f21z;
      s_syx += y12f * f21x;
      s_syy += y12f * f21y;
      s_syz += y12f * f21z;
      s_szx += z12f * f21x;
      s_szy += z12f * f21y;
      s_szz += z12f * f21z;

      if (blend == 0.0f) {
        // Pure nuclear core: do not evaluate singular vdW or exponential IWP.
        g_f12x[index] = g_f12y[index] = g_f12z[index] = 0.0f;
        continue;
      }
      
      // calc rep
      float C = ilp_para.C[type1][type2];
      float lambda_ = ilp_para.lambda[type1][type2];
      float delta2inv = ilp_para.delta2inv[type1][type2];
      float epsilon = ilp_para.epsilon[type1][type2];
      float z0 = ilp_para.z0[type1][type2];
      // calc_rep
      float prodnorm1, rhosq1, rdsq1, exp0, exp1, frho1, Erep, Vilp;
      float fpair, fpair1, fsum, delx, dely, delz, fkcx, fkcy, fkcz;
      float dprodnorm1[3] = {0.0f, 0.0f, 0.0f};
      float fp1[3] = {0.0f, 0.0f, 0.0f};
      float fprod1[3] = {0.0f, 0.0f, 0.0f};
      float fk[3] = {0.0f, 0.0f, 0.0f};

      delx = -x12f;
      dely = -y12f;
      delz = -z12f;

      float delx_half = delx * 0.5f;
      float dely_half = dely * 0.5f;
      float delz_half = delz * 0.5f;

      // calculate the transverse distance
      prodnorm1 = normal[0] * delx + normal[1] * dely + normal[2] * delz;
      rhosq1 = rsq - prodnorm1 * prodnorm1;
      rdsq1 = rhosq1 * delta2inv;

      // store exponents
      exp0 = expf(-lambda_ * (r - z0));
      exp1 = expf(-rdsq1);

      frho1 = exp1 * C;
      Erep = 0.5f * epsilon + frho1;
      Vilp = exp0 * Erep;

      // derivatives
      fpair = lambda_ * exp0 * rinv * Erep;
      fpair1 = 2.0f * exp0 * frho1 * delta2inv;
      fsum = fpair + fpair1;

      float prodnorm1_m_fpair1 = prodnorm1 * fpair1;
      float Vilp_m_dTap_m_rinv = Vilp * dTap * rinv;

      // derivatives of the product of rij and ni, the result is a vector
      dprodnorm1[0] = 
        dnormdri[0][0] * delx + dnormdri[1][0] * dely + dnormdri[2][0] * delz;
      dprodnorm1[1] = 
        dnormdri[0][1] * delx + dnormdri[1][1] * dely + dnormdri[2][1] * delz;
      dprodnorm1[2] = 
        dnormdri[0][2] * delx + dnormdri[1][2] * dely + dnormdri[2][2] * delz;
      fp1[0] = prodnorm1_m_fpair1 * normal[0];
      fp1[1] = prodnorm1_m_fpair1 * normal[1];
      fp1[2] = prodnorm1_m_fpair1 * normal[2];
      fprod1[0] = prodnorm1_m_fpair1 * dprodnorm1[0];
      fprod1[1] = prodnorm1_m_fpair1 * dprodnorm1[1];
      fprod1[2] = prodnorm1_m_fpair1 * dprodnorm1[2];

      fkcx = (delx * fsum - fp1[0]) * Tap - Vilp_m_dTap_m_rinv * delx;
      fkcy = (dely * fsum - fp1[1]) * Tap - Vilp_m_dTap_m_rinv * dely;
      fkcz = (delz * fsum - fp1[2]) * Tap - Vilp_m_dTap_m_rinv * delz;

      if (blend != 1.0f || blend_p != 0.0f) {
        const float switch_radial = blend_p*Tap*Vilp*rinv;
        fkcx = blend*fkcx - switch_radial*delx;
        fkcy = blend*fkcy - switch_radial*dely;
        fkcz = blend*fkcz - switch_radial*delz;
      }

      s_fx += fkcx - blend*fprod1[0] * Tap;
      s_fy += fkcy - blend*fprod1[1] * Tap;
      s_fz += fkcz - blend*fprod1[2] * Tap;

      g_f12x[index] = fkcx;
      g_f12y[index] = fkcy;
      g_f12z[index] = fkcz;

      float minus_prodnorm1_m_fpair1_m_Tap = -prodnorm1 * fpair1 * Tap * blend;
      for (int kk = 0; kk < ilp_neighbor_number; ++kk) {
        // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
        dprodnorm1[0] = dnormal[0][kk][0] * delx + dnormal[1][kk][0] * dely +
            dnormal[2][kk][0] * delz;
        dprodnorm1[1] = dnormal[0][kk][1] * delx + dnormal[1][kk][1] * dely +
            dnormal[2][kk][1] * delz;
        dprodnorm1[2] = dnormal[0][kk][2] * delx + dnormal[1][kk][2] * dely +
            dnormal[2][kk][2] * delz;
        fk[0] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[0];
        fk[1] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[1];
        fk[2] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[2];

        g_f12x_ilp_neigh[n1 + number_of_particles * kk] += fk[0];
        g_f12y_ilp_neigh[n1 + number_of_particles * kk] += fk[1];
        g_f12z_ilp_neigh[n1 + number_of_particles * kk] += fk[2];

        s_sxx += delkix_half[kk] * fk[0];
        s_sxy += delkix_half[kk] * fk[1];
        s_sxz += delkix_half[kk] * fk[2];
        s_syx += delkiy_half[kk] * fk[0];
        s_syy += delkiy_half[kk] * fk[1];
        s_syz += delkiy_half[kk] * fk[2];
        s_szx += delkiz_half[kk] * fk[0];
        s_szy += delkiz_half[kk] * fk[1];
        s_szz += delkiz_half[kk] * fk[2];
      }
      s_pe += blend * Tap * Vilp;
      s_sxx += delx_half * fkcx;
      s_sxy += delx_half * fkcy;
      s_sxz += delx_half * fkcz;
      s_syx += dely_half * fkcx;
      s_syy += dely_half * fkcy;
      s_syz += dely_half * fkcz;
      s_szx += delz_half * fkcx;
      s_szy += delz_half * fkcy;
      s_szz += delz_half * fkcz;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;

    // save potential
    g_potential[n1] += s_pe;

  }
}

// build a neighbor list for reducing force
static __global__ void build_reduce_neighbor_list(
  const int number_of_particles,
  const int N1,
  const int N2,
  const int *g_neighbor_number,
  const int *g_neighbor_list,
  int *g_reduce_neighbor_list)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int l, r, m, tmp_value;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + i1 * number_of_particles;
      int n2 = g_neighbor_list[index];

      l = 0;
      r = g_neighbor_number[n2];
      while (l < r) {
        m = (l + r) >> 1;
        tmp_value = g_neighbor_list[n2 + number_of_particles * m];
        if (tmp_value < n1) {
          l = m + 1;
        } else if (tmp_value > n1) {
          r = m - 1;
        } else {
          break;
        }
      }
      g_reduce_neighbor_list[index] = (l + r) >> 1;
    }
  }
}

// reduce the rep force
static __global__ void scatter_iwp_normal_force(
  const int N, const int N1, const int N2, const Box box,
  const int* normal_number, const int* normal_list,
  const double* x, const double* y, const double* z,
  const float* partial_fx, const float* partial_fy, const float* partial_fz,
  double* fx, double* fy, double* fz, double* virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (i >= N2) return;
  const int offset[3][3] = {{0, 3, 4}, {6, 1, 5}, {7, 8, 2}};
  for (int k = 0; k < normal_number[i]; ++k) {
    const int index = i + N * k;
    const int j = normal_list[index];
    const double f[3] = {partial_fx[index], partial_fy[index], partial_fz[index]};
    atomicAdd(fx + j, f[0]);
    atomicAdd(fy + j, f[1]);
    atomicAdd(fz + j, f[2]);
    double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
    apply_mic(box, dx, dy, dz);
    const double dr[3] = {dx, dy, dz};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        atomicAdd(virial + j + N * offset[a][b], 0.5 * dr[a] * f[b]);
      }
    }
  }
}

static __global__ void reduce_force_many_body(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int *g_neighbor_number,
  const int *g_neighbor_list,
  int *g_reduce_neighbor_list,
  int *g_ilp_neighbor_number,
  int *g_ilp_neighbor_list,
  const double *__restrict__ g_x,
  const double *__restrict__ g_y,
  const double *__restrict__ g_z,
  double *g_fx,
  double *g_fy,
  double *g_fz,
  double *g_virial,
  float *g_f12x,
  float *g_f12y,
  float *g_f12z,
  float *g_f12x_ilp_neigh,
  float *g_f12y_ilp_neigh,
  float *g_f12z_ilp_neigh)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  float s_fx = 0.0f;                                   // force_x
  float s_fy = 0.0f;                                   // force_y
  float s_fz = 0.0f;                                   // force_z
  float s_sxx = 0.0f;                                  // virial_stress_xx
  float s_sxy = 0.0f;                                  // virial_stress_xy
  float s_sxz = 0.0f;                                  // virial_stress_xz
  float s_syx = 0.0f;                                  // virial_stress_yx
  float s_syy = 0.0f;                                  // virial_stress_yy
  float s_syz = 0.0f;                                  // virial_stress_yz
  float s_szx = 0.0f;                                  // virial_stress_zx
  float s_szy = 0.0f;                                  // virial_stress_zy
  float s_szz = 0.0f;                                  // virial_stress_zz


  if (n1 < N2) {
    double x12d, y12d, z12d;
    float x12f, y12f, z12f;
    int neighbor_number_1 = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    // calculate energy and force
    for (int i1 = 0; i1 < neighbor_number_1; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_neighbor_list[index];

      x12d = g_x[n2] - x1;
      y12d = g_y[n2] - y1;
      z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);
      x12f = float(x12d);
      y12f = float(y12d);
      z12f = float(z12d);

      index = n2 + number_of_particles * g_reduce_neighbor_list[index];
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

      s_fx -= f21x;
      s_fy -= f21y;
      s_fz -= f21z;

      // per-atom virial
      s_sxx += x12f * f21x * 0.5f;
      s_sxy += x12f * f21y * 0.5f;
      s_sxz += x12f * f21z * 0.5f;
      s_syx += y12f * f21x * 0.5f;
      s_syy += y12f * f21y * 0.5f;
      s_syz += y12f * f21z * 0.5f;
      s_szx += z12f * f21x * 0.5f;
      s_szy += z12f * f21y * 0.5f;
      s_szz += z12f * f21z * 0.5f;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
  }
}


// ----- NEP part -----
static __global__ void find_descriptor(
  const int* nep_map,
  const int* type_map,
  const int* labels,
  void* h_parambs,
  void* h_annmbs,
  const int total_types,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[labels[n1]];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float* paramb = (float*)h_parambs + nep_id * IWP_M6X6_H_PAR_OFFSET;
    int* paramb_int = (int*) paramb;
    int type_offset = nep_id * total_types;
    int t1 = type_map[g_type[n1] + type_offset];
    float* annmb = (float*)h_annmbs + nep_id * IWP_M6X6_H_ANN_OFFSET;
    int* atomic_numbers = IWP_M6X6_INT_PTR(paramb + IWP_M6X6_PTRAN);
    float* c = IWP_M6X6_FLT_PTR(annmb + IWP_M6X6_PTRC);
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = type_map[g_type[n2] + type_offset];
      float rc = paramb[IWP_M6X6_RCR];

      if (paramb_int[IWP_M6X6_UTC]) {
        rc = min(
          (COVALENT_RADIUS[atomic_numbers[t1]] +
           COVALENT_RADIUS[atomic_numbers[t2]]) *
            paramb[IWP_M6X6_TCRF],
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb_int[IWP_M6X6_BSR], rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb_int[IWP_M6X6_NMAXR]; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb_int[IWP_M6X6_BSR]; ++k) {
          int c_index = get_c_index(
            t1 * paramb_int[IWP_M6X6_NT] + t2, n, k, paramb_int[IWP_M6X6_NMAXR], paramb_int[IWP_M6X6_BSR]);
          gn12 += fn12[k] * c[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    // for (int n = 0; n <= paramb.n_max_angular; ++n) {
    for (int n = 0; n <= paramb_int[IWP_M6X6_NMAXA]; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        double x12double = g_x[n2] - x1;
        double y12double = g_y[n2] - y1;
        double z12double = g_z[n2] - z1;
        apply_mic(box, x12double, y12double, z12double);
        float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = type_map[g_type[n2] + type_offset];
        float rc = paramb[IWP_M6X6_RCA];

        if (paramb_int[IWP_M6X6_UTC]) {
          rc = min(
            (COVALENT_RADIUS[atomic_numbers[t1]] +
             COVALENT_RADIUS[atomic_numbers[t2]]) *
              paramb[IWP_M6X6_TCAF],
            rc);
        }
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb_int[IWP_M6X6_BSA], rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb_int[IWP_M6X6_BSA]; ++k) {
          int c_index = get_c_index(
            t1 * paramb_int[IWP_M6X6_NT] + t2,
            n,
            k,
            paramb_int[IWP_M6X6_NMAXA],
            paramb_int[IWP_M6X6_BSA],
            paramb_int[IWP_M6X6_NCR]);
          gn12 += fn12[k] * c[c_index];
        }
        accumulate_s(paramb_int[IWP_M6X6_LMAX], d12, x12, y12, z12, gn12, s);
      }
      // find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      find_q(paramb_int[IWP_M6X6_LMAX], paramb_int[IWP_M6X6_NUML], paramb_int[IWP_M6X6_NMAXA] + 1, n, s, q + (paramb_int[IWP_M6X6_NMAXR] + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    // normalize descriptor
    float* q_scaler = IWP_M6X6_FLT_PTR(paramb + IWP_M6X6_PTRQS);
    int ann_dim = *((int*)annmb + IWP_M6X6_ANNDIM);
    for (int d = 0; d < ann_dim; ++d) {
      q[d] = q[d] * q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    int ann_num_neurons1 = *((int*)annmb + IWP_M6X6_NNEUR);
    apply_ann_one_layer(
      ann_dim,
      ann_num_neurons1,
      IWP_M6X6_FLT_PTR(annmb + IWP_M6X6_PTRW0) + t1 * ann_dim * ann_num_neurons1,
      IWP_M6X6_FLT_PTR(annmb + IWP_M6X6_PTRB0) + t1 * ann_num_neurons1,
      IWP_M6X6_FLT_PTR(annmb + IWP_M6X6_PTRW1) + t1 * ann_num_neurons1,
      &annmb[IWP_M6X6_OUTB1],
      q,
      F,
      Fp);
    g_pe[n1] += F;

    // for (int d = 0; d < annmb.dim; ++d) {
    for (int d = 0; d < ann_dim; ++d) {
      // g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
      g_Fp[d * N + n1] = Fp[d] * q_scaler[d];
    }

    // set ptrs to null
    paramb = nullptr;
    paramb_int = nullptr;
    annmb = nullptr;
    atomic_numbers = nullptr;
    q_scaler = nullptr;
    c = nullptr;
  }
}

static __global__ void find_force_radial(
  const int* nep_map,
  const int* type_map,
  const int* labels,
  void* h_parambs,
  void* h_annmbs,
  const int total_types,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[labels[n1]];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    float* paramb = (float*)h_parambs + nep_id * IWP_M6X6_H_PAR_OFFSET;
    int* paramb_int = (int*) paramb;
    float* annmb = (float*)h_annmbs + nep_id * IWP_M6X6_H_ANN_OFFSET;
    int* atomic_numbers = IWP_M6X6_INT_PTR(paramb + IWP_M6X6_PTRAN);
    float* c = IWP_M6X6_FLT_PTR(annmb + IWP_M6X6_PTRC);
    int type_offset = nep_id * total_types;
    int t1 = type_map[g_type[n1] + type_offset];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = type_map[g_type[n2] + type_offset];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      float fc12, fcp12;

      float rc = paramb[IWP_M6X6_RCR];
      if (paramb_int[IWP_M6X6_UTC]) {
        rc = min(
          (COVALENT_RADIUS[atomic_numbers[t1]] +
           COVALENT_RADIUS[atomic_numbers[t2]]) *
            paramb[IWP_M6X6_TCRF],
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb_int[IWP_M6X6_BSR], rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb_int[IWP_M6X6_NMAXR]; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb_int[IWP_M6X6_BSR]; ++k) {
          int c_index_12 = get_c_index(
            t1 * paramb_int[IWP_M6X6_NT] + t2, n, k, paramb_int[IWP_M6X6_NMAXR], paramb_int[IWP_M6X6_BSR]);
          int c_index_21 = get_c_index(
            t2 * paramb_int[IWP_M6X6_NT] + t1, n, k, paramb_int[IWP_M6X6_NMAXR], paramb_int[IWP_M6X6_BSR]);
          gnp12 += fnp12[k] * c[c_index_12];
          gnp21 += fnp12[k] * c[c_index_21];
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];

      s_sxx += r12[0] * f21[0];
      s_syy += r12[1] * f21[1];
      s_szz += r12[2] * f21[2];
      s_sxy += r12[0] * f21[1];
      s_sxz += r12[0] * f21[2];
      s_syx += r12[1] * f21[0];
      s_syz += r12[1] * f21[2];
      s_szx += r12[2] * f21[0];
      s_szy += r12[2] * f21[1];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;

    // set ptrs to null
    paramb = nullptr;
    paramb_int = nullptr;
    annmb = nullptr;
    atomic_numbers = nullptr;
    c = nullptr;
  }
}

static __global__ void find_partial_force_angular(
  const int* nep_map,
  const int* type_map,
  const int* labels,
  void* h_parambs,
  void* h_annmbs,
  const int total_types,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[labels[n1]];
    float* paramb = (float*)h_parambs + nep_id * IWP_M6X6_H_PAR_OFFSET;
    int* paramb_int = (int*) paramb;
    float* annmb = (float*)h_annmbs + nep_id * IWP_M6X6_H_ANN_OFFSET;
    int* atomic_numbers = IWP_M6X6_INT_PTR(paramb + IWP_M6X6_PTRAN);
    float* c = IWP_M6X6_FLT_PTR(annmb + IWP_M6X6_PTRC);

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb_int[IWP_M6X6_DIMA]; ++d) {
      Fp[d] = g_Fp[(paramb_int[IWP_M6X6_NMAXR] + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb_int[IWP_M6X6_NMAXA] + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int type_offset = nep_id * total_types;
    int t1 = type_map[g_type[n1] + type_offset];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = type_map[g_type[n2] + type_offset];
      float rc = paramb[IWP_M6X6_RCA];
      if (paramb_int[IWP_M6X6_UTC]) {
        rc = min(
          (COVALENT_RADIUS[atomic_numbers[t1]] +
           COVALENT_RADIUS[atomic_numbers[t2]]) *
            paramb[IWP_M6X6_TCAF],
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb_int[IWP_M6X6_BSA], rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb_int[IWP_M6X6_NMAXA]; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb_int[IWP_M6X6_BSA]; ++k) {
          int c_index = get_c_index(
            t1 * paramb_int[IWP_M6X6_NT] + t2,
            n,
            k,
            paramb_int[IWP_M6X6_NMAXA],
            paramb_int[IWP_M6X6_BSA],
            paramb_int[IWP_M6X6_NCR]);
          gn12 += fn12[k] * c[c_index];
          gnp12 += fnp12[k] * c[c_index];
        }
        accumulate_f12(paramb_int[IWP_M6X6_LMAX], paramb_int[IWP_M6X6_NUML], n, paramb_int[IWP_M6X6_NMAXA] + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }

    // set ptrs to null
    paramb = nullptr;
    paramb_int = nullptr;
    annmb = nullptr;
    atomic_numbers = nullptr;
    c = nullptr;
  }
}

static __global__ void gpu_find_force_many_body_nep(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  float s_fx = 0.0f;  // force_x
  float s_fy = 0.0f;  // force_y
  float s_fz = 0.0f;  // force_z
  float s_sxx = 0.0f; // virial_stress_xx
  float s_sxy = 0.0f; // virial_stress_xy
  float s_sxz = 0.0f; // virial_stress_xz
  float s_syx = 0.0f; // virial_stress_yx
  float s_syy = 0.0f; // virial_stress_yy
  float s_syz = 0.0f; // virial_stress_yz
  float s_szx = 0.0f; // virial_stress_zx
  float s_szy = 0.0f; // virial_stress_zy
  float s_szz = 0.0f; // virial_stress_zz

  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);

      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];
      int offset = 0;
      for (int k = 0; k < neighbor_number_2; ++k) {
        if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = offset * number_of_particles + n2;
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

      // per atom force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // per-atom virial
      s_sxx += x12 * f21x;
      s_syy += y12 * f21y;
      s_szz += z12 * f21z;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
  }
}

#define BLOCK_SIZE_FORCE_NEP 64
void iwp_m6x6_find_properties_many_body_nep(
  Box& box,
  int N1,
  int N2,
  const int* NN,
  const int* NL,
  const float* f12x,
  const float* f12y,
  const float* f12z,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = position_per_atom.size() / 3;
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE_NEP + 1;

  gpu_find_force_many_body_nep<<<grid_size, BLOCK_SIZE_FORCE_NEP>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    NN,
    NL,
    f12x,
    f12y,
    f12z,
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}

static __global__ void init_f12(
  const int N,
  float *g_f12x_ilp,
  float *g_f12y_ilp,
  float *g_f12z_ilp,
  float *g_f12x_ilp_neigh,
  float *g_f12y_ilp_neigh,
  float *g_f12z_ilp_neigh,
  const int *NN) 
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    #pragma unroll
    for (int i = 0; i < IWP_M6X6_MAX_ILP_NEIGHBOR_TMD; ++i) {
      g_f12x_ilp_neigh[n1 + N * i] = 0.0f;
      g_f12y_ilp_neigh[n1 + N * i] = 0.0f;
      g_f12z_ilp_neigh[n1 + N * i] = 0.0f;
    }

    for (int i = 0; i < NN[n1]; ++i) {
      g_f12x_ilp[n1 + N * i] = 0.0f;
      g_f12y_ilp[n1 + N * i] = 0.0f;
      g_f12z_ilp[n1 + N * i] = 0.0f;
    }


  }
}


// nep part of compute func

// define the pure virtual func
void IWP_NEP_M6X6::compute(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom)
{
  // nothing
}


// #define CHECK_NEIGHBOR 1
#define BLOCK_SIZE_ILP 128
//#define USE_FIXED_NEIGHBOR 1
#define UPDATE_TEMP 10
#define BIG_ILP_CUTOFF_SQUARE 50.0
// find force and related quantities
void IWP_NEP_M6X6::compute_ilp(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom,
  const std::vector<Group> &group)
{

  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_ILP + 1;

  if (ilp_group_method < 0 || ilp_group_method >= group.size() ||
      nep_group_method < 0 || nep_group_method >= group.size()) {
    PRINT_INPUT_ERROR("IWP/NEP group method is absent from model.xyz.\n");
  }
  if (group[nep_group_method].size.size() != nep_map_cpu.size()) {
    PRINT_INPUT_ERROR("NEP map group count differs from model.xyz.\n");
  }
  // get labels of IWP and NEP
  const int *group_label_ilp = group[ilp_group_method].label.data();
  const int *group_label_nep = group[nep_group_method].label.data();
  int* g_nep_map = nep_map.data();
  int* g_type_map = type_map.data();
  const int total_types = type_map_cpu.size() / num_nep;


#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    iwp_m6x6_find_neighbor_ilp_nep(
      N1,
      N2,
      g_nep_map,
      g_type_map,
      group_label_nep,
      h_parambs,
      rc,
      box,
      group_label_ilp,
      type,
      position_per_atom,
      ilp_data.cell_count,
      ilp_data.cell_count_sum,
      ilp_data.cell_contents,
      ilp_data.NN,
      ilp_data.NL,
      nep_data.NN_radial,
      nep_data.NL_radial,
      nep_data.NN_angular,
      nep_data.NL_angular, ilp_para, iwp_normal_sign_gpu.data(),
      ilp_data.ilp_NN, ilp_data.ilp_NL, max_iwp_normal_rc, max_nep_rc);
    
  gpu_sort_neighbor_list<<<number_of_atoms, min(1024, max_MN_radial), max_MN_radial * sizeof(int)>>>(
    number_of_atoms, nep_data.NN_radial.data(), nep_data.NL_radial.data());
  GPU_CHECK_KERNEL

  gpu_sort_neighbor_list<<<number_of_atoms, min(1024, max_MN_angular), max_MN_angular * sizeof(int)>>>(
    number_of_atoms, nep_data.NN_angular.data(), nep_data.NL_angular.data());
  GPU_CHECK_KERNEL


    build_reduce_neighbor_list<<<grid_size, BLOCK_SIZE_ILP>>>(
      number_of_atoms,
      N1,
      N2,
      ilp_data.NN.data(),
      ilp_data.NL.data(),
      ilp_data.reduce_NL.data());
  GPU_CHECK_KERNEL
#ifdef USE_FIXED_NEIGHBOR
  }
  num_calls %= UPDATE_TEMP;
#endif

  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + number_of_atoms;
  const double* z = position_per_atom.data() + number_of_atoms * 2;
  const int *NN = ilp_data.NN.data();
  const int *NL = ilp_data.NL.data();
  int *reduce_NL = ilp_data.reduce_NL.data();
  int *ilp_NL = ilp_data.ilp_NL.data();
  int *ilp_NN = ilp_data.ilp_NN.data();

  double *g_fx = force_per_atom.data();
  double *g_fy = force_per_atom.data() + number_of_atoms;
  double *g_fz = force_per_atom.data() + number_of_atoms * 2;
  double *g_virial = virial_per_atom.data();
  double *g_potential = potential_per_atom.data();
  float *g_f12x = ilp_data.f12x.data();
  float *g_f12y = ilp_data.f12y.data();
  float *g_f12z = ilp_data.f12z.data();
  float *g_f12x_ilp_neigh = ilp_data.f12x_ilp_neigh.data();
  float *g_f12y_ilp_neigh = ilp_data.f12y_ilp_neigh.data();
  float *g_f12z_ilp_neigh = ilp_data.f12z_ilp_neigh.data();

  // initialize partial force
  init_f12<<<grid_size, BLOCK_SIZE_ILP>>>(
    number_of_atoms,
    g_f12x, g_f12y, g_f12z, 
    g_f12x_ilp_neigh, g_f12y_ilp_neigh, g_f12z_ilp_neigh, NN);
  GPU_CHECK_KERNEL


  gpu_find_force<<<grid_size, BLOCK_SIZE_ILP>>>(
    ilp_para,
    number_of_atoms,
    N1,
    N2,
    box,
    NN,
    NL,
    ilp_NN,
    ilp_NL,
    group_label_ilp,
    type.data(),
    x,
    y,
    z,
    g_fx,
    g_fy,
    g_fz,
    g_virial,
    g_potential,
    g_f12x,
    g_f12y,
    g_f12z,
    g_f12x_ilp_neigh,
    g_f12y_ilp_neigh,
    g_f12z_ilp_neigh,
    iwp_normal_sign_gpu.data());
  GPU_CHECK_KERNEL


  scatter_iwp_normal_force<<<grid_size, BLOCK_SIZE_ILP>>>(
    number_of_atoms, N1, N2, box, ilp_NN, ilp_NL, x, y, z,
    g_f12x_ilp_neigh, g_f12y_ilp_neigh, g_f12z_ilp_neigh,
    g_fx, g_fy, g_fz, g_virial);
  GPU_CHECK_KERNEL

  reduce_force_many_body<<<grid_size, BLOCK_SIZE_ILP>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    NN,
    NL,
    reduce_NL,
    ilp_NN,
    ilp_NL,
    x,
    y,
    z,
    g_fx,
    g_fy,
    g_fz,
    g_virial,
    g_f12x,
    g_f12y,
    g_f12z,
    g_f12x_ilp_neigh,
    g_f12y_ilp_neigh,
    g_f12z_ilp_neigh);
    GPU_CHECK_KERNEL

  // compute NEP
  const int BLOCK_SIZE_NEP = 64;
  const int N = type.size();
  const int grid_size_nep = (N2 - N1 - 1) / BLOCK_SIZE_NEP + 1;


  static int num_calls_n = 0;
  if (num_calls_n++ % 1000 == 0) {
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < nep_data.cpu_NN_radial[n]) {
        radial_actual = nep_data.cpu_NN_radial[n];
      }
      if (angular_actual < nep_data.cpu_NN_angular[n]) {
        angular_actual = nep_data.cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls_n - 1 << ": "
                << "radial(max=" << max_MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << max_MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor<<<grid_size_nep, BLOCK_SIZE_NEP>>>(
    g_nep_map,
    g_type_map,
    group_label_nep,
    h_parambs,
    h_annmbs,
    total_types,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  find_force_radial<<<grid_size_nep, BLOCK_SIZE_NEP>>>(
    g_nep_map,
    g_type_map,
    group_label_nep,
    h_parambs,
    h_annmbs,
    total_types,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size_nep, BLOCK_SIZE_NEP>>>(
    g_nep_map,
    g_type_map,
    group_label_nep,
    h_parambs,
    h_annmbs,
    total_types,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data());
  GPU_CHECK_KERNEL

  iwp_m6x6_find_properties_many_body_nep(
    box,
    N1,
    N2,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    position_per_atom,
    force_per_atom,
    virial_per_atom);
  GPU_CHECK_KERNEL

}
