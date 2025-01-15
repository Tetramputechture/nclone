#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class Laser : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 23;
  static constexpr float RADIUS = 5.9f;
  static constexpr float SPIN_SPEED = 0.010471975f; // roughly 2pi/600
  static constexpr float SURFACE_FLAT_SPEED = 0.1f;
  static constexpr float SURFACE_CORNER_SPEED = 0.005524805665672641f; // roughly 0.1/(5.9*pi)

  Laser(Simulation *sim, float xcoord, float ycoord, int orientation, int mode);

  void think() override;
  bool isThinkable() const override { return true; }
  std::vector<float> getState(bool minimalState = false) const override;

private:
  void thinkSpinner();
  void thinkSurface();

  int orientation;
  int mode;
  float angle = 0.0f;
  float xstart;
  float ystart;
  float xend;
  float yend;
  float progress = 0.0f;
  bool clockwise = false;
};