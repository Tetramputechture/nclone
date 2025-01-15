#pragma once

#include "../entity.hpp"

class EntityGold : public Entity
{
public:
  static constexpr int entityType = 2;
  static constexpr float RADIUS = 6.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 8192;

  EntityGold(int entityType, Simulation *sim, float xcoord, float ycoord);

  void logicalCollision() override;
};