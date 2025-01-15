#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class BounceBlock : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 17;
  static constexpr float SEMI_SIDE = 9.0f;
  static constexpr float STIFFNESS = 0.02222222222222222f;
  static constexpr float DAMPENING = 0.98f;
  static constexpr float STRENGTH = 0.2f;
  static constexpr int MAX_COUNT_PER_LEVEL = 512;

  BounceBlock(Simulation *sim, float xcoord, float ycoord);

  void move() override;
  void physicalCollision() override;
  void logicalCollision() override;
  bool isMovable() const override { return true; }

private:
  float xspeedOld = 0.0f;
  float yspeedOld = 0.0f;
};