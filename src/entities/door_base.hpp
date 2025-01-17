#pragma once

#include "entity.hpp"
#include "../physics/segment.hpp"
#include <memory>
#include <vector>
#include <utility>

class DoorBase : public Entity
{
public:
  DoorBase(int type, Simulation *sim, float xcoord, float ycoord, int orientation, float swXcoord, float swYcoord);
  virtual ~DoorBase() = default;

  void logicalCollision() override;
  bool isLogicalCollidable() const override { return true; }
  std::vector<float> getState(bool minimalState = false) const override;

protected:
  void changeState(bool closed);
  bool closed = true;
  int orientation;
  float swXcoord;
  float swYcoord;
  bool isVertical;
  std::shared_ptr<Segment> segment;
  std::vector<std::pair<int, int>> gridEdges;
};