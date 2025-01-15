#include "door_base.hpp"
#include "../simulation.hpp"

EntityDoorBase::EntityDoorBase(int entityType, Simulation *sim, float xcoord, float ycoord,
                               int orientation, float swXcoord, float swYcoord)
    : Entity(entityType, sim, xcoord, ycoord), orientation(orientation), swXcoord(swXcoord), swYcoord(swYcoord)
{
  initSegment();
}

void EntityDoorBase::initSegment()
{
  float x1 = xpos;
  float y1 = ypos;
  float x2 = xpos;
  float y2 = ypos;

  // Adjust segment endpoints based on orientation
  if (orientation == 0 || orientation == 4)
  {
    x2 += 24.0f;
  }
  else
  {
    y2 += 24.0f;
  }

  segment = std::make_shared<Segment>(x1, y1, x2, y2);
  sim->getSegmentsAt(cell).push_back(segment);
}

void EntityDoorBase::changeState(bool newClosed)
{
  if (closed != newClosed)
  {
    closed = newClosed;
    segment->active = closed;
  }
}

std::vector<float> EntityDoorBase::getState(bool minimalState) const
{
  auto baseState = Entity::getState(minimalState);
  if (!minimalState)
  {
    baseState.push_back(static_cast<float>(orientation));
    baseState.push_back(swXcoord);
    baseState.push_back(swYcoord);
    baseState.push_back(closed ? 1.0f : 0.0f);
  }
  return baseState;
}