#include "door_locked.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

DoorLocked::DoorLocked(Simulation *sim, float xcoord, float ycoord,
                       int orientation, float swXcoord, float swYcoord)
    : DoorBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, swXcoord, swYcoord)
{
}

void DoorLocked::logicalCollision()
{
  if (!active)
    return;

  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    active = false;
    changeState(false);
    sim->getNinja()->doorsOpened++;
    logCollision();
  }
}