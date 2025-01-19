#include "exit_door.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

ExitDoor::ExitDoor(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
}

std::optional<EntityCollisionResult> ExitDoor::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    ninja->win();
  }
  return std::nullopt;
}
