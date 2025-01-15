#pragma once

#include "entity.hpp"

class EntityToggleMine : public Entity
{
public:
  static constexpr int entityType = 1;                              // Also handles type 21 for toggled state
  static constexpr std::array<float, 3> RADII = {4.0f, 3.5f, 4.5f}; // 0:toggled, 1:untoggled, 2:toggling
  static constexpr int MAX_COUNT_PER_LEVEL = 8192;

  EntityToggleMine(int entityType, Simulation *sim, float xcoord, float ycoord, int state);

  void think() override;
  void logicalCollision() override;
  bool isThinkable() const override { return true; }

  void setState(int newState);
  std::vector<float> getState(bool minimalState = false) const override;

private:
  int state; // 0:toggled, 1:untoggled, 2:toggling
  int toggleTimer = 0;
};
