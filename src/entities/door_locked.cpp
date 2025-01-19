#include "door_locked.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

DoorLocked::DoorLocked(Simulation *sim, float xcoord, float ycoord,
                       int orientation, float swXcoord, float swYcoord)
    : DoorBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, swXcoord, swYcoord)
{
}

std::optional<EntityCollisionResult> DoorLocked::logicalCollision()
{
  if (!active)
    return std::nullopt;

  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    setActive(false);
    changeState(false);
    sim->getNinja()->doorsOpened++;
    logCollision();
  }
  return std::nullopt;
}
