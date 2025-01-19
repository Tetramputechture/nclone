#include "door_trap.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

DoorTrap::DoorTrap(Simulation *sim, float xcoord, float ycoord,
                   int orientation, float swXcoord, float swYcoord)
    : DoorBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, swXcoord, swYcoord)
{
}

EntityCollisionResult DoorTrap::logicalCollision()
{
  if (!active)
    return EntityCollisionResult::noCollision();

  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    active = false;
    changeState(true);
    logCollision();
    return EntityCollisionResult::logicalCollision();
  }
  return EntityCollisionResult::noCollision();
}