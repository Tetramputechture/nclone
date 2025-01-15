#include "door_regular.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

EntityDoorRegular::EntityDoorRegular(Simulation *sim, float xcoord, float ycoord,
                                     int orientation, float swXcoord, float swYcoord)
    : EntityDoorBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, swXcoord, swYcoord)
{
}

void EntityDoorRegular::think()
{
  if (!active)
    return;

  changeState(!ninjaInRange);
  ninjaInRange = false;
}

void EntityDoorRegular::logicalCollision()
{
  if (!active)
    return;

  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    ninjaInRange = true;
    logCollision;
  }
}