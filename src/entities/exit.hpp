#pragma once

#include "entity.hpp"

class EntityExit : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 3;
  static constexpr float RADIUS = 12.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 16;

  EntityExit(int entityType, Simulation *sim, float xcoord, float ycoord);

  void logicalCollision() override;
};