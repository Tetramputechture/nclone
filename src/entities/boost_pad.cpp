#include "boost_pad.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

BoostPad::BoostPad(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
}

EntityCollisionResult BoostPad::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return EntityCollisionResult::noCollision();

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    ninja->xspeed *= BOOST;
    ninja->yspeed *= BOOST;
    return EntityCollisionResult::logicalCollision();
  }
  return EntityCollisionResult::noCollision();
}