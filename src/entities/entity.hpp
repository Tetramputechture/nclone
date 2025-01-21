#pragma once

#include <memory>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <array>
#include <optional>
#include "entity_collision_result.hpp"

// Forward declaration
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
  virtual std::optional<EntityCollisionResult> physicalCollision() { return std::nullopt; }
  virtual std::optional<EntityCollisionResult> logicalCollision() { return std::nullopt; }

  // State getters/setters
  virtual std::vector<float> getState(bool minimalState = false) const;
  void gridMove();
  void logCollision(int state = 1);
  void logPosition();

  // Virtual methods for entity state
  virtual bool isActive() const { return active; }
  virtual bool isMovable() const { return false; }
  virtual bool isThinkable() const { return false; }
  virtual bool isLogicalCollidable() const { return false; }
  virtual bool isPhysicalCollidable() const { return false; }
  virtual int getType() const { return type; }
  virtual std::pair<int, int> getCell() const { return cell; }

  // Getters
  int getEntityType() const { return entityType; }
  Simulation *getSimulation() const { return sim; }
  float getXPos() const { return xpos; }
  float getYPos() const { return ypos; }
  float getXSpeed() const { return xspeed; }
  float getYSpeed() const { return yspeed; }
  float getXPosOld() const { return xposOld; }
  float getYPosOld() const { return yposOld; }

  // Setters
  void setEntityType(int type) { entityType = type; }
  void setSimulation(Simulation *simulation) { sim = simulation; }
  void setXPos(float x) { xpos = x; }
  void setYPos(float y) { ypos = y; }
  void setXSpeed(float xs) { xspeed = xs; }
  void setYSpeed(float ys) { yspeed = ys; }
  void setXPosOld(float x) { xposOld = x; }
  void setYPosOld(float y) { yposOld = y; }
  void setActive(bool isActive) { active = isActive; }

protected:
  // Protected member variables for derived classes
  int entityType;
  Simulation *sim;
  float xpos;
  float ypos;
  float xspeed = 0.0f;
  float yspeed = 0.0f;
  float xposOld;
  float yposOld;
  std::vector<std::tuple<int, float, float>> posLog;
  std::vector<std::tuple<int, float, float>> speedLog;
  std::vector<int> collisionLog;
  bool active = true;
  bool logPositions = false;
  bool logCollisions = true;
  int type = 0;
  std::pair<int, int> cell;
  int lastExportedState = -1;
  int lastExportedFrame = -1;
  std::pair<float, float> lastExportedCoords;
  std::vector<int> exportedChunks;

  std::pair<int, int> calculateCell() const;
};