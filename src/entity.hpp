#pragma once

#include <memory>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "physics/physics.hpp"

class Simulation;

class Entity
{
public:
  // Static members
  static std::array<int, 40> entityCounts;

  // Constructor
  Entity(int entityType, Simulation *sim, float xcoord, float ycoord);
  virtual ~Entity() = default;

  // Virtual methods that can be overridden by derived classes
  virtual void think() {}
  virtual void move() {}
  virtual void physicalCollision() {}
  virtual void logicalCollision() {}
  virtual bool isMovable() const { return false; }
  virtual bool isThinkable() const { return false; }

  // State getters/setters
  virtual std::vector<float> getState(bool minimalState = false) const;
  std::pair<int, int> getCell() const;
  void gridMove();
  void logCollision(int state = 1);
  void logPosition();

  // Public member variables
  int entityType;
  Simulation *sim;
  float xpos;
  float ypos;
  float xspeed = 0.0f;
  float yspeed = 0.0f;
  float xposOld;
  float yposOld;
  bool active = true;
  std::pair<int, int> cell;

protected:
  // Protected member variables for derived classes
  std::vector<std::tuple<int, float, float>> posLog;
  std::vector<std::tuple<int, float, float>> speedLog;
  std::vector<int> collisionLog;
};