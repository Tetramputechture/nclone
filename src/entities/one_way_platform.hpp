#pragma once

#include "entity.hpp"
#include "../simulation.hpp"
#include <optional>

class OneWayPlatform : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 11;
  static constexpr float SEMI_SIDE = 12.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 512;

  OneWayPlatform(Simulation *sim, float xcoord, float ycoord, int orientation);

  EntityCollisionResult physicalCollision() override;
  EntityCollisionResult logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }
  bool isPhysicalCollidable() const override { return true; }

  std::optional<std::tuple<std::pair<float, float>, std::pair<float, float>>>
  calculateDepenetration(const Ninja *ninja) const;

private:
  int orientation;
  float normalX;
  float normalY;
};