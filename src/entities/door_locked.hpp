#pragma once

#include "door_base.hpp"

class EntityDoorLocked : public EntityDoorBase
{
public:
  static constexpr int entityType = 6;
  static constexpr float RADIUS = 5.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;

  EntityDoorLocked(int entityType, Simulation *sim, float xcoord, float ycoord,
                   int orientation, float swXcoord, float swYcoord);

  void logicalCollision() override;
};