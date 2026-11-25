/*
  Radial local normal for M6X6 nanowires.

  For atom i, use the four nearest unlike atoms in the same wire within
  pair-specific rcut_normal. The caller builds this list independently of NEP:

    u_k = (r_k-r_i)/|r_k-r_i|
    v_i = s_i * sum_k u_k
    n_i = v_i/|v_i|

  The returned derivatives follow the IWP_NEP_M6X6 tensor convention:
    dnorm_center[a][b]       = d n_a / d r_i,b
    dnorm_neighbors[a][k][b] = d n_a / d r_k,b
*/
#pragma once
#include <cmath>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define IWP_HD __host__ __device__ __forceinline__
#else
#define IWP_HD inline
#endif

namespace iwp_m6x6 {

constexpr int NORMAL_NEIGHBORS = 4;
constexpr int NUM_SUPPORTED_ELEMENTS = 5;
inline int atomic_number(const char* element)
{
  if (std::strcmp(element, "Mo") == 0) return 42;
  if (std::strcmp(element, "W") == 0) return 74;
  if (std::strcmp(element, "S") == 0) return 16;
  if (std::strcmp(element, "Se") == 0) return 34;
  if (std::strcmp(element, "Te") == 0) return 52;
  return 0;
}

// C2 interpolation: IWP has weight w, uncut universal ZBL has weight 1-w.
// Callers skip the IWP expression entirely when w == 0 (never 0*infinity).
template <typename T>
IWP_HD void zbl_blend_weight(const T r, const T inner, const T outer, T& w, T& wp)
{
  w = T(1);
  wp = T(0);
  if (outer <= T(0) || r >= outer) return;
  if (r <= inner) {
    w = T(0);
    return;
  }
  const T inv_width = T(1) / (outer - inner);
  const T x = (r - inner) * inv_width;
  const T y = T(1) - x;
  // Complementary form avoids cancellation of 1-w near the outer endpoint.
  w = x <= T(0.5) ? x*x*x*(T(10)+x*(T(-15)+T(6)*x))
                   : T(1)-y*y*y*(T(10)+y*(T(-15)+T(6)*y));
  wp = T(30)*x*x*y*y*inv_width;
}

// e in eV, de=d(e)/dr in eV/angstrom; zizj includes the Coulomb constant.
template <typename T>
IWP_HD void zbl_core(const T r, const T zizj, const T a_inv, T& e, T& de)
{
  const T x = r*a_inv;
  const T p1 = T(0.18175)*exp(-T(3.1998)*x);
  const T p2 = T(0.50986)*exp(-T(0.94229)*x);
  const T p3 = T(0.28022)*exp(-T(0.40290)*x);
  const T p4 = T(0.02817)*exp(-T(0.20162)*x);
  const T phi = p1+p2+p3+p4;
  const T phi_p = -T(3.1998)*p1-T(0.94229)*p2-T(0.40290)*p3-T(0.20162)*p4;
  const T inv_r = T(1)/r;
  e = zizj*phi*inv_r;
  de = zizj*inv_r*(phi_p*a_inv-phi*inv_r);
}

inline float normal_sign(const char* element)
{
  if (std::strcmp(element, "Mo") == 0 || std::strcmp(element, "W") == 0) return +1.0f;
  if (std::strcmp(element, "S") == 0 || std::strcmp(element, "Se") == 0 ||
      std::strcmp(element, "Te") == 0) return -1.0f;
  return 0.0f;
}

template <int Storage>
IWP_HD bool calc_radial_normal_and_derivatives(
  const float (&vectors)[Storage][3],
  const int count,
  const float sign,
  float (&normal)[3],
  float (&dnorm_center)[3][3],
  float (&dnorm_neighbors)[3][Storage][3])
{
  static_assert(Storage >= NORMAL_NEIGHBORS, "IWP normal storage must hold four neighbors.");

  for (int a = 0; a < 3; ++a) {
    normal[a] = 0.0f;
    for (int b = 0; b < 3; ++b) {
      dnorm_center[a][b] = 0.0f;
      for (int k = 0; k < Storage; ++k) dnorm_neighbors[a][k][b] = 0.0f;
    }
  }

  if (count != NORMAL_NEIGHBORS || sign == 0.0f) return false;

  float unit[NORMAL_NEIGHBORS][3];
  float inv_distance[NORMAL_NEIGHBORS];
  float vector_sum[3] = {0.0f, 0.0f, 0.0f};

  for (int k = 0; k < NORMAL_NEIGHBORS; ++k) {
    const float x = vectors[k][0];
    const float y = vectors[k][1];
    const float z = vectors[k][2];
    const float distance_sq = x * x + y * y + z * z;
    if (distance_sq <= 1.0e-20f) return false;

    inv_distance[k] = 1.0f / sqrtf(distance_sq);
    unit[k][0] = x * inv_distance[k];
    unit[k][1] = y * inv_distance[k];
    unit[k][2] = z * inv_distance[k];
    for (int a = 0; a < 3; ++a) vector_sum[a] += sign * unit[k][a];
  }

  const float vector_norm_sq =
    vector_sum[0] * vector_sum[0] +
    vector_sum[1] * vector_sum[1] +
    vector_sum[2] * vector_sum[2];
  if (vector_norm_sq <= 1.0e-20f) return false;

  const float inv_vector_norm = 1.0f / sqrtf(vector_norm_sq);
  for (int a = 0; a < 3; ++a) normal[a] = vector_sum[a] * inv_vector_norm;

  // P_ab = (delta_ab - n_a n_b) / |v|
  float projection[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      projection[a][b] = (((a == b) ? 1.0f : 0.0f) - normal[a] * normal[b]) * inv_vector_norm;
    }
  }

  for (int k = 0; k < NORMAL_NEIGHBORS; ++k) {
    // A_cb = d u_c / d d_b = (delta_cb-u_c u_b)/|d|
    float unit_derivative[3][3];
    for (int c = 0; c < 3; ++c) {
      for (int b = 0; b < 3; ++b) {
        unit_derivative[c][b] =
          (((c == b) ? 1.0f : 0.0f) - unit[k][c] * unit[k][b]) * inv_distance[k];
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        float derivative = 0.0f;
        for (int c = 0; c < 3; ++c) {
          derivative += projection[a][c] * sign * unit_derivative[c][b];
        }
        dnorm_neighbors[a][k][b] = derivative;
        dnorm_center[a][b] -= derivative;
      }
    }
  }

  return true;
}

} // namespace iwp_m6x6

#undef IWP_HD

#if defined(__CUDACC__) || defined(__HIPCC__)
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

#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <stdio.h>
#include <cstdint>
#include <vector>

// M6X6: Mo W S Se Te; four radial normal neighbors
#define IWP_M6X6_MAX_TYPE_ILP_NEP 5
#define IWP_M6X6_CUDA_MAX_NL_ILP_NEP_CBN 1024   // neighs in different layer
#define IWP_M6X6_CUDA_MAX_NL_ILP_NEP_TMD 2048  // neighs in different layer
#define IWP_M6X6_MAX_ILP_NEIGHBOR_CBN 3 // radial normal storage
#define IWP_M6X6_MAX_ILP_NEIGHBOR_TMD 4 // radial normal storage
#define IWP_M6X6_MAX_BIG_ILP_NEIGHBOR_CBN 128



// this nep data struct is just for ILP+NEP potential
struct IWP_M6X6_NEP_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<char> para_buffer_gpu;
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
};


struct IWP_M6X6_Para {
  float rcutsq_ilp[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];
  float d[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];
  float d_Seff[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];      // d / S_R / r_eff
  float C_6[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];
  float z0[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];          // beta
  float lambda[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];      // alpha / beta
  float epsilon[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];
  float C[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];
  float delta2inv[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];   // 1 / delta ^ 2
  float S[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];           // scale
  float rcut_global[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP];           // scale
  // Optional unordered-pair short-range replacement; outer==0 disables it.
  float zbl_inner[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP] = {};
  float zbl_outer[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP] = {};
  float zbl_zizj[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP] = {};
  float zbl_a_inv[IWP_M6X6_MAX_TYPE_ILP_NEP][IWP_M6X6_MAX_TYPE_ILP_NEP] = {};
};

struct IWP_M6X6_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> reduce_NL;
  GPU_Vector<int> ilp_NN, ilp_NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<float> f12x;
  GPU_Vector<float> f12y;
  GPU_Vector<float> f12z;
  GPU_Vector<float> f12x_ilp_neigh;
  GPU_Vector<float> f12y_ilp_neigh;
  GPU_Vector<float> f12z_ilp_neigh;
};

// define paramb memory location
#define IWP_M6X6_UTC             0     // use_typewise_cutoff
#define IWP_M6X6_TCRF            1     // typewise_cutoff_radial_factor
#define IWP_M6X6_TCAF            2     // typewise_cutoff_angular_factor
#define IWP_M6X6_VERSION         3     // nep version
#define IWP_M6X6_RCR             4     // rc_radial
#define IWP_M6X6_RCA             5     // rc_angular     
#define IWP_M6X6_RCIR            6     // rcinv_radial
#define IWP_M6X6_RCIA            7     // rcinv_angular
#define IWP_M6X6_MNR             8     // MN_radial
#define IWP_M6X6_MNA             9     // MN_angular
#define IWP_M6X6_NMAXR           10    // n_max_radial
#define IWP_M6X6_NMAXA           11    // n_max_angular
#define IWP_M6X6_LMAX            12    // L_max
#define IWP_M6X6_DIMA            13    // dim_angular
#define IWP_M6X6_NUML            14    // num_L
#define IWP_M6X6_BSR             15    // basis_size_radial
#define IWP_M6X6_BSA             16    // basis_size_angular
#define IWP_M6X6_NTS             17    // num_types_sq
#define IWP_M6X6_NCR             18    // num_c_radial
#define IWP_M6X6_NT              19    // num_types
#define IWP_M6X6_PTRQS           20    // pointer of q_scaler
#define IWP_M6X6_PTRAN           (IWP_M6X6_PTRQS + IWP_M6X6_PTR_OFFSET)  // pointer of atomic number
#define IWP_M6X6_H_PAR_OFFSET    (IWP_M6X6_PTRAN + IWP_M6X6_PTR_OFFSET)  // ptr offset of head of paramb            

// define annmb memory location
#define IWP_M6X6_SIZEOF_POINTER  (UINTPTR_MAX / 255 % 255)  // size of pointer
#define IWP_M6X6_SIZEOF_INT      4     // size of int
#define IWP_M6X6_PTR_OFFSET      (IWP_M6X6_SIZEOF_POINTER / IWP_M6X6_SIZEOF_INT) // ptr offset by int ptr
#define IWP_M6X6_ANNDIM          0     // ann dim
#define IWP_M6X6_NNEUR           1     // num_neurous1
#define IWP_M6X6_OUTB1           2     // bias for output layer
#define IWP_M6X6_EPT             3     // empty
#define IWP_M6X6_PTRC            4     // pointer of c
#define IWP_M6X6_PTRW0           (IWP_M6X6_PTRC + IWP_M6X6_PTR_OFFSET)   // pointer of w0 of type0
#define IWP_M6X6_PTRB0           (IWP_M6X6_PTRW0 + IWP_M6X6_PTR_OFFSET)  // pointer of b0 of type0
#define IWP_M6X6_PTRW1           (IWP_M6X6_PTRB0 + IWP_M6X6_PTR_OFFSET)  // pointer of w1 of type0
#define IWP_M6X6_H_ANN_OFFSET    (IWP_M6X6_PTRW1 + IWP_M6X6_PTR_OFFSET)  // ptr offset of head of annmb

#define IWP_M6X6_INT_PTR(m)      ((int*) *((uintptr_t*) (m)))
#define IWP_M6X6_FLT_PTR(m)      ((float*) *((uintptr_t*) (m)))

class IWP_NEP_M6X6 : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    float typewise_cutoff_radial_factor = 0.0f;
    float typewise_cutoff_angular_factor = 0.0f;
    int version = 4; // NEP version
    int model_type =
      0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    float q_scaler[140];
    int atomic_numbers[NUM_ELEMENTS];
  };


  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
  };


  using Potential::compute;
  IWP_NEP_M6X6(FILE*, FILE*, int, int, bool require_zbl = false);
  virtual ~IWP_NEP_M6X6(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  
  virtual void compute_ilp(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    const std::vector<Group>& group) override;
  

protected:
  IWP_M6X6_Para ilp_para;
  IWP_M6X6_Data ilp_data;

private:
  int num_nep = 0; // number of NEP file
  int max_MN_angular = 0;     // max of parambs[i].MN_angular
  int max_MN_radial = 0;      // max of parambs[i].MN_radial
  int max_n_max_angular = 0;  // max of parambs[i].n_max_angular
  int max_dim = 0;            // max of annmbs[i].dim
  std::vector<ParaMB> parambs;
  std::vector<ANN> annmbs;
  char* h_parambs;                  // pointer to the head of parambs in gpu
  char* h_annmbs;                   // pointer to the head of annmbs in gpu
  std::vector<int> nep_map_cpu;     // map nep group to nep parameters (cpu)
  GPU_Vector<int> nep_map;          // map nep group to nep parameters (gpu)
  std::vector<int> type_map_cpu;    // map ilp type to nep type (cpu)
  GPU_Vector<int> type_map;         // map ilp type to nep type (gpu)
  IWP_M6X6_NEP_Data nep_data;

  // One wire group method and one NEP map group method
  int ilp_group_method = 0;
  int nep_group_method = 0;

  // M = Mo/W (+1), X = S/Se/Te (-1).
  float iwp_normal_sign_cpu[IWP_M6X6_MAX_TYPE_ILP_NEP] = {0.0f};
  GPU_Vector<float> iwp_normal_sign_gpu;
  double max_iwp_normal_rc = 0.0;

  // save max nep rcut to make sure pbc
  double max_nep_rc;
};

static __constant__ float IWP_M6X6_Tap_coeff[8];

#endif // CUDA or HIP compiler
