#include "door_regular.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

DoorRegular::DoorRegular(Simulation *sim, float xcoord, float ycoord,
                         int orientation, float swXcoord, float swYcoord)
    : DoorBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, swXcoord, swYcoord)
{
}

void DoorRegular::think()
{
  if (!active)
    return;

  if (!ninjaInRange)
  {
    openTimer++;
    if (openTimer > 5)
    {
      changeState(true); // close the door
    }
  }
  ninjaInRange = false;
}

EntityCollisionResult DoorRegular::logicalCollision()
{
  if (!active)
    return EntityCollisionResult::noCollision();

  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    ninjaInRange = true;
    openTimer = 0;
    changeState(false); // open the door
    return EntityCollisionResult::logicalCollision();
  }
  return EntityCollisionResult::noCollision();
}
