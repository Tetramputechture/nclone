#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class Thwump : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 20;
  static constexpr float SEMI_SIDE = 9.0f;
  static constexpr float FORWARD_SPEED = 20.0f / 7.0f;
  static constexpr float BACKWARD_SPEED = 8.0f / 7.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 128;

  Thwump(Simulation *sim, float xcoord, float ycoord, int orientation);

  void think() override;
  void move() override;
  EntityCollisionResult physicalCollision() override;
  EntityCollisionResult logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }
  bool isPhysicalCollidable() const override { return true; }
  bool isMovable() const override { return true; }
  bool isThinkable() const override { return true; }

  void setState(int state);

private:
  int orientation;
  int state = 0; // 0: idle, 1: charging, 2: returning
  float xstart;
  float ystart;
  float speed = 0.0f;
};