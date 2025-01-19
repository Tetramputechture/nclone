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

  ShoveThwump(Simulation *sim, float xcoord, float ycoord);

  void think() override;
  void move() override;
  std::optional<EntityCollisionResult> physicalCollision() override;
  std::optional<EntityCollisionResult> logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }
  bool isPhysicalCollidable() const override { return true; }
  bool isMovable() const override { return true; }
  bool isThinkable() const override { return true; }

private:
  float xorigin;
  float yorigin;
  float xdir = 0.0f;
  float ydir = 0.0f;
  bool activated = false;
  int state = 0; // 0:immobile, 1:activated, 2:launching, 3:retreating

  void setState(int newState);
  bool moveIfPossible(float xdir, float ydir, float speed);
};