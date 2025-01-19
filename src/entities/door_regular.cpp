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
  if (closed)
    return;

  openTimer++;
  if (openTimer > 5)
    changeState(true);
}

std::optional<EntityCollisionResult> DoorRegular::logicalCollision()
{
  if (Physics::overlapCircleVsCircle(
          swXcoord, swYcoord, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    changeState(false);
    openTimer = 0;
  }
  return std::nullopt;
}
