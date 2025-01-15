#include "drone_zap.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

DroneZap::DroneZap(Simulation *sim, float xcoord, float ycoord, int orientation, int mode)
    : DroneBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, mode, DEFAULT_SPEED)
{
}

void DroneZap::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  if (Physics::overlapCircleVsCircle(xpos, ypos, RADIUS, ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    ninja->kill(0, xpos, ypos, 0.0f, 0.0f);
  }
}