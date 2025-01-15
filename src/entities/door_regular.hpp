#pragma once

#include "door_base.hpp"

class EntityDoorRegular : public EntityDoorBase
{
public:
  static constexpr int entityType = 5;
  static constexpr float RADIUS = 10.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;

  EntityDoorRegular(int entityType, Simulation *sim, float xcoord, float ycoord,
                    int orientation, float swXcoord, float swYcoord);

  void think() override;
  void logicalCollision() override;
  bool isThinkable() const override { return true; }

private:
  bool ninjaInRange = false;
};