#pragma once

#include "entity.hpp"
#include "../simulation.hpp"
#include <array>
#include <unordered_map>

class DroneBase : public Entity
{
public:
  static constexpr float RADIUS = 7.5f;
  static constexpr float GRID_WIDTH = 24.0f;

  DroneBase(int entityType, Simulation *sim, float xcoord, float ycoord, int orientation, int mode, float speed);

  void move() override;
  bool isMovable() const override { return true; }
  bool isThinkable() const override { return true; }
  std::vector<float> getState(bool minimalState = false) const override;

protected:
  void turn(int dir);
  virtual void chooseNextDirectionAndGoal();
  bool testNextDirectionAndGoal(int dir);

  int orientation;
  int mode;
  float speed;
  float goalX;
  float goalY;

private:
  static const std::unordered_map<int, std::array<float, 2>> DIR_TO_VEC;
  static const std::unordered_map<int, std::array<int, 4>> DIR_LIST;
};