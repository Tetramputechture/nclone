#include "shove_thwump.hpp"

ShoveThwump::ShoveThwump(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), xstart(xcoord), ystart(ycoord)
{
}

void ShoveThwump::think()
{
  // TODO: Implement think logic
}

void ShoveThwump::move()
{
  // TODO: Implement move logic
}

void ShoveThwump::physicalCollision()
{
  // TODO: Implement physical collision
}

void ShoveThwump::logicalCollision()
{
  // TODO: Implement logical collision
}