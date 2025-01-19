#pragma once

#include "entity.hpp"

class Gold : public Entity
{
public:
  Gold(Simulation *sim, float xcoord, float ycoord);
  EntityCollisionResult logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }

  static constexpr int ENTITY_TYPE = 2;
  static constexpr float RADIUS = 6.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 8192;
};