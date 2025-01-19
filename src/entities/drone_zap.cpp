#include "drone_zap.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

DroneZap::DroneZap(Simulation *sim, float xcoord, float ycoord, int orientation, int mode)
    : DroneBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, mode, SPEED)
{
}

std::optional<EntityCollisionResult> DroneZap::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja->isValidTarget())
    return std::nullopt;

  if (Physics::overlapCircleVsCircle(xpos, ypos, RADIUS, ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    ninja->kill(0, 0, 0, 0, 0);
  }
  return std::nullopt;
}