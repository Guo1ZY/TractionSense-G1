#include <cassert>
#include <cmath>
#include <iostream>

#include "hall_foot_model.h"

namespace {

double Norm(const std::array<float, 3>& value) {
  return std::sqrt(
      static_cast<double>(value[0]) * value[0] +
      static_cast<double>(value[1]) * value[1] +
      static_cast<double>(value[2]) * value[2]);
}

}  // namespace

int main() {
  g1_hall::HallFootForwardModel model;
  g1_hall::FootContacts empty;
  auto output = model.Update(0.02, empty);
  for (const auto& foot : output) {
    for (const auto& sensor : foot) {
      assert(Norm(sensor) < 1.0e-9);
    }
  }

  // P02 local XY from the shared normalized layout.
  const double p02_x = 0.035 + 0.282175 * 0.21502;
  const double p02_y = -0.015712 * 0.08004;
  g1_hall::FootContacts normal;
  normal[0].push_back({{p02_x, p02_y}, {{0.0, 0.0, 300.0}}});
  for (int step = 0; step < 12; ++step) {
    output = model.Update(0.02, normal);
  }
  const double near_response = Norm(output[0][2]);
  const double far_response = Norm(output[0][14]);
  assert(std::isfinite(near_response));
  assert(near_response > 1.0e-4);
  assert(near_response > far_response);
  assert(Norm(output[1][2]) < 1.0e-8);
  assert(model.maximum_compression_m() > 0.0);

  g1_hall::FootContacts shear = normal;
  shear[0][0].force_local_n = {{90.0, -45.0, 300.0}};
  const auto before_shear = output[0][2];
  for (int step = 0; step < 12; ++step) {
    output = model.Update(0.02, shear);
  }
  assert(
      std::fabs(output[0][2][0] - before_shear[0]) > 1.0e-5 ||
      std::fabs(output[0][2][1] - before_shear[1]) > 1.0e-5);

  const double loaded = Norm(output[0][2]);
  for (int step = 0; step < 200; ++step) {
    output = model.Update(0.02, empty);
  }
  assert(Norm(output[0][2]) < 0.05 * loaded);
  for (const auto& foot : output) {
    for (const auto& sensor : foot) {
      for (float axis : sensor) {
        assert(std::isfinite(axis));
        assert(std::fabs(axis) <= 6.0f);
      }
    }
  }

  std::cout << "MuJoCo Hall forward model tests passed\n";
  return 0;
}
