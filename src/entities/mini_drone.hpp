#pragma once

#include "drone_base.hpp"

class MiniDrone : public DroneBase
{
public:
  static constexpr int ENTITY_TYPE = 26;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;
  static constexpr float DEFAULT_SPEED = 20.0f / 7.0f;

  MiniDrone(Simulation *sim, float xcoord, float ycoord, int orientation, int mode);

  void logicalCollision() override;
};