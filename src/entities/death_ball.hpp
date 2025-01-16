#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class DeathBall : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 25;
  static constexpr float RADIUS = 6.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 128;

  DeathBall(Simulation *sim, float xcoord, float ycoord);

  void think() override;
  void logicalCollision() override;
  bool isThinkable() const override { return true; }
};