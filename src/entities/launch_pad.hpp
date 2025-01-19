#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class LaunchPad : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 10;
  static constexpr float RADIUS = 6.0f;
  static constexpr float BOOST = 36.0f / 7.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;

  LaunchPad(Simulation *sim, float xcoord, float ycoord, int orientation);

  std::optional<EntityCollisionResult> logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }

private:
  int orientation;
  float normalX, normalY;
};