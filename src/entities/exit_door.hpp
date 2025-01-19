#pragma once

#include "entity.hpp"

class ExitDoor : public Entity
{
public:
  ExitDoor(Simulation *sim, float xcoord, float ycoord);
  std::optional<EntityCollisionResult> logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }

  static constexpr int ENTITY_TYPE = 3;
  static constexpr float RADIUS = 12.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 16;
};