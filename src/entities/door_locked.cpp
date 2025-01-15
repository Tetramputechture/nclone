#include "door_locked.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

EntityDoorLocked::EntityDoorLocked(int entityType, Simulation *sim, float xcoord, float ycoord,
                                   int orientation, float swXcoord, float swYcoord)
    : EntityDoorBase(entityType, sim, xcoord, ycoord, orientation, swXcoord, swYcoord)
{
}

void EntityDoorLocked::logicalCollision()
{
  if (!active)
    return;

  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    active = false;
    changeState(false);
    sim->getNinja()->doors_opened++;
    logCollision();
  }
}