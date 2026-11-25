#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <cstdio>
#include <cstdlib>
#include "../../src/force/iwp_nep_m6x6.cuh"

namespace {
constexpr int Storage = 6;

void require_close(float actual, float expected, float tolerance, const char* label)
{
  if (std::fabs(actual - expected) > tolerance) {
    std::fprintf(stderr, "%s: actual=% .9e expected=% .9e diff=% .9e tol=% .9e\n",
                 label, actual, expected, std::fabs(actual - expected), tolerance);
    std::exit(EXIT_FAILURE);
  }
}

bool normal_only(const float (&vectors)[Storage][3], float sign, float (&normal)[3])
{
  float center[3][3];
  float neighbors[3][Storage][3];
  return iwp_m6x6::calc_radial_normal_and_derivatives<Storage>(
    vectors, iwp_m6x6::NORMAL_NEIGHBORS, sign, normal, center, neighbors);
}
}

int main()
{
  for (const char* element : {"Mo", "W"}) {
    require_close(iwp_m6x6::normal_sign(element), +1.0f, 0.0f, "metal sign");
  }
  for (const char* element : {"S", "Se", "Te"}) {
    require_close(iwp_m6x6::normal_sign(element), -1.0f, 0.0f, "chalcogen sign");
  }
  for (const char* element : {"C", "B", "N", "O", "H", "mo"}) {
    require_close(iwp_m6x6::normal_sign(element), 0.0f, 0.0f, "unsupported element");
  }

  float vectors[Storage][3] = {
    { 1.20f,  0.15f,  0.35f},
    {-0.30f,  1.05f,  0.42f},
    {-1.10f, -0.18f,  0.27f},
    { 0.22f, -0.92f,  0.58f},
    { 0.00f,  0.00f,  0.00f},
    { 0.00f,  0.00f,  0.00f}
  };

  float normal[3];
  float center[3][3];
  float neighbors[3][Storage][3];
  if (!iwp_m6x6::calc_radial_normal_and_derivatives<Storage>(
        vectors, iwp_m6x6::NORMAL_NEIGHBORS, +1.0f, normal, center, neighbors)) {
    std::fprintf(stderr, "normal calculation unexpectedly failed\n");
    return EXIT_FAILURE;
  }

  float sum[3] = {0.0f, 0.0f, 0.0f};
  for (int k = 0; k < iwp_m6x6::NORMAL_NEIGHBORS; ++k) {
    const float norm = std::sqrt(
      vectors[k][0] * vectors[k][0] +
      vectors[k][1] * vectors[k][1] +
      vectors[k][2] * vectors[k][2]);
    for (int a = 0; a < 3; ++a) sum[a] += vectors[k][a] / norm;
  }
  const float sum_norm = std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
  for (int a = 0; a < 3; ++a) require_close(normal[a], sum[a] / sum_norm, 2.0e-6f, "direct normal");

  float reversed[3];
  if (!normal_only(vectors, -1.0f, reversed)) return EXIT_FAILURE;
  for (int a = 0; a < 3; ++a) require_close(reversed[a], -normal[a], 2.0e-6f, "sign reversal");

  const float h = 2.0e-4f;
  const float derivative_tolerance = 2.5e-3f;
  for (int k = 0; k < iwp_m6x6::NORMAL_NEIGHBORS; ++k) {
    for (int b = 0; b < 3; ++b) {
      float plus_vectors[Storage][3];
      float minus_vectors[Storage][3];
      std::copy(&vectors[0][0], &vectors[0][0] + Storage * 3, &plus_vectors[0][0]);
      std::copy(&vectors[0][0], &vectors[0][0] + Storage * 3, &minus_vectors[0][0]);
      plus_vectors[k][b] += h;
      minus_vectors[k][b] -= h;
      float plus[3], minus[3];
      if (!normal_only(plus_vectors, +1.0f, plus) || !normal_only(minus_vectors, +1.0f, minus)) {
        return EXIT_FAILURE;
      }
      for (int a = 0; a < 3; ++a) {
        const float finite_difference = (plus[a] - minus[a]) / (2.0f * h);
        require_close(neighbors[a][k][b], finite_difference, derivative_tolerance, "neighbor derivative");
      }
    }
  }

  for (int b = 0; b < 3; ++b) {
    float plus_vectors[Storage][3];
    float minus_vectors[Storage][3];
    std::copy(&vectors[0][0], &vectors[0][0] + Storage * 3, &plus_vectors[0][0]);
    std::copy(&vectors[0][0], &vectors[0][0] + Storage * 3, &minus_vectors[0][0]);
    for (int k = 0; k < iwp_m6x6::NORMAL_NEIGHBORS; ++k) {
      plus_vectors[k][b] -= h;
      minus_vectors[k][b] += h;
    }
    float plus[3], minus[3];
    if (!normal_only(plus_vectors, +1.0f, plus) || !normal_only(minus_vectors, +1.0f, minus)) {
      return EXIT_FAILURE;
    }
    for (int a = 0; a < 3; ++a) {
      const float finite_difference = (plus[a] - minus[a]) / (2.0f * h);
      require_close(center[a][b], finite_difference, derivative_tolerance, "center derivative");

      float translation_sum = center[a][b];
      for (int k = 0; k < iwp_m6x6::NORMAL_NEIGHBORS; ++k) translation_sum += neighbors[a][k][b];
      require_close(translation_sum, 0.0f, 2.0e-6f, "translation invariance");
    }
  }

  float invalid_vectors[Storage][3] = {};
  if (normal_only(invalid_vectors, +1.0f, normal)) {
    std::fprintf(stderr, "zero-length vectors should be rejected\n");
    return EXIT_FAILURE;
  }

  std::puts("PASS: IWP radial normal and analytical derivatives");
  return EXIT_SUCCESS;
}

