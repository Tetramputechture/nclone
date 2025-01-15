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
  sim->get_segments_at(cell).push_back(segment);
}

void EntityDoorBase::changeState(bool new_closed)
{
  if (closed != new_closed)
  {
    closed = new_closed;
    segment->active = closed;
  }
}

std::vector<float> EntityDoorBase::getState(bool minimal_state) const
{
  auto base_state = Entity::getState(minimal_state);
  if (!minimal_state)
  {
    base_state.push_back(static_cast<float>(orientation));
    base_state.push_back(swXcoord);
    base_state.push_back(swYcoord);
    base_state.push_back(closed ? 1.0f : 0.0f);
  }
  return base_state;
}