#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class ShoveThwump : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 28;
  static constexpr float SEMI_SIDE = 12.0f;
  static constexpr float RADIUS = 8.0f; // for the projectile inside
  static constexpr int MAX_COUNT_PER_LEVEL = 128;
  static constexpr float FORWARD_SPEED = 20.0f / 7.0f;
  static constexpr float BACKWARD_SPEED = 8.0f / 7.0f;

  ShoveThwump(Simulation *sim, float xcoord, float ycoord);

  void think() override;
  void move() override;
  void physicalCollision() override;
  void logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }
  bool isPhysicalCollidable() const override { return true; }
  bool isMovable() const override { return true; }
  bool isThinkable() const override { return true; }

private:
  float xstart;
  float ystart;
  float speed = 0.0f;
  int state = 0; // 0: idle, 1: charging, 2: returning
  void setState(int newState);
};