#pragma once

#include "door_base.hpp"

class DoorLocked : public DoorBase
{
public:
  static constexpr int ENTITY_TYPE = 6;
  static constexpr float RADIUS = 5.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;

  DoorLocked(Simulation *sim, float xcoord, float ycoord,
             int orientation, float swXcoord, float swYcoord);

  EntityCollisionResult logicalCollision() override;
};