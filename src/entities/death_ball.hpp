#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class DeathBall : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 25;
  static constexpr float RADIUS = 5.0f;
  static constexpr float RADIUS2 = 8.0f;
  static constexpr float ACCELERATION = 0.04f;
  static constexpr float MAX_SPEED = 0.85f;
  static constexpr float DRAG_MAX_SPEED = 0.9f;
  static constexpr float DRAG_NO_TARGET = 0.95f;
  static constexpr int MAX_COUNT_PER_LEVEL = 64;

  DeathBall(Simulation *sim, float xcoord, float ycoord);

  void think() override;
  std::optional<EntityCollisionResult> logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }
  bool isThinkable() const override { return true; }
};