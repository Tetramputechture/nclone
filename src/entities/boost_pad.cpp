#include "boost_pad.hpp"

BoostPad::BoostPad(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
}

void BoostPad::logicalCollision()
{
  // TODO: Implement logical collision
}