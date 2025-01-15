#include "door_trap.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

EntityDoorTrap::EntityDoorTrap(int entityType, Simulation *sim, float xcoord, float ycoord,
                               int orientation, float swXcoord, float swYcoord)
    : EntityDoorBase(entityType, sim, xcoord, ycoord, orientation, swXcoord, swYcoord)
{
}

void EntityDoorTrap::logicalCollision()
{
  if (!active)
    return;

  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    active = false;
    changeState(true);
    logCollision();
  }
}