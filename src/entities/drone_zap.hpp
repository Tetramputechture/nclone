#pragma once

#include "drone_base.hpp"

class DroneZap : public DroneBase
{
public:
  static constexpr int ENTITY_TYPE = 14;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;
  static constexpr float SPEED = 8.0f / 7.0f;

  DroneZap(Simulation *sim, float xcoord, float ycoord, int orientation, int mode);
  std::optional<std::pair<float, float>> logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }
};