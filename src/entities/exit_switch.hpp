#pragma once

#include "entity.hpp"
#include "exit_door.hpp"

class ExitSwitch : public Entity
{
public:
  ExitSwitch(Simulation *sim, float xcoord, float ycoord, ExitDoor *parent);
  void logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }

  static constexpr int ENTITY_TYPE = 4;
  static constexpr float RADIUS = 6.0f;

private:
  ExitDoor *parent;
};
