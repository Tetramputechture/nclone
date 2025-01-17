#pragma once

#include "door_base.hpp"

class DoorRegular : public DoorBase
{
public:
  static constexpr int ENTITY_TYPE = 5;
  static constexpr float RADIUS = 10.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;

  DoorRegular(Simulation *sim, float xcoord, float ycoord,
              int orientation, float swXcoord, float swYcoord);

  void think() override;
  void logicalCollision() override;
  bool isThinkable() const override { return true; }

private:
  bool ninjaInRange = false;
  int openTimer = 0;
};