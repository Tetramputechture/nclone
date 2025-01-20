#pragma once

#include "entity.hpp"

class ToggleMine : public Entity
{
public:
  static constexpr int ENTITY_TYPE = 1;                             // Also handles type 21 for toggled state
  static constexpr std::array<float, 3> RADII = {4.0f, 3.5f, 4.5f}; // 0:toggled, 1:untoggled, 2:toggling
  static constexpr int MAX_COUNT_PER_LEVEL = 8192;

  ToggleMine(Simulation *sim, float xcoord, float ycoord, int state);

  void think() override;
  std::optional<EntityCollisionResult> logicalCollision() override;
  bool isThinkable() const override { return true; }
  bool isLogicalCollidable() const override { return true; }

  void setState(int newState);
  std::vector<float> getState(bool minimalState = false) const override;
  float getRadius() const { return RADII[state]; }

private:
  int state; // 0:toggled, 1:untoggled, 2:toggling
  bool activated = false;
};
