#pragma once

#include "entity.hpp"
#include "../simulation.hpp"

class BoostPad : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 24;
  static constexpr float RADIUS = 6.0f;
  static constexpr int MAX_COUNT_PER_LEVEL = 128;

  BoostPad(Simulation *sim, float xcoord, float ycoord);

  void move() override;
  bool isMovable() const override { return true; }

private:
  bool isTouchingNinja;
};