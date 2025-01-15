#pragma once

#include "entity.hpp"
#include <memory>
#include "../physics/segment.hpp"

class EntityDoorBase : public Entity
{
public:
  EntityDoorBase(int entityType, Simulation *sim, float xcoord, float ycoord,
                 int orientation, float swXcoord, float swYcoord);

  virtual ~EntityDoorBase() = default;

  void changeState(bool closed);
  std::vector<float> getState(bool minimalState = false) const override;

protected:
  int orientation;
  float swXcoord;
  float swYcoord;
  bool closed = true;
  std::shared_ptr<Segment> segment;

  void initSegment();
};