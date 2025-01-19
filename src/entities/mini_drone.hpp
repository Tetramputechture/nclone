#pragma once

#include "drone_base.hpp"

class MiniDrone : public DroneBase
{
public:
  MiniDrone(Simulation *sim, float xcoord, float ycoord, int orientation, int mode);
  std::optional<EntityCollisionResult> logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }

  static constexpr int ENTITY_TYPE = 26;
  static constexpr float RADIUS = 4.0f;
  static constexpr float GRID_WIDTH = 12.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 512;
};