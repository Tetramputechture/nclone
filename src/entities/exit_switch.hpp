#pragma once

#include "../entity.hpp"

class EntityExit;

class EntityExitSwitch : public Entity
{
public:
  static constexpr int entityType = 4;
  static constexpr float RADIUS = 6.0f;

  EntityExitSwitch(int entityType, Simulation *sim, float xcoord, float ycoord, EntityExit *parent);

  void logicalCollision() override;

private:
  EntityExit *parent;
};