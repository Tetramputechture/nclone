#include "mini_drone.hpp"
#include "../physics/physics.hpp"

MiniDrone::MiniDrone(Simulation *sim, float xcoord, float ycoord, int orientation, int mode)
    : DroneBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, mode, 1.3f)
{
}

std::optional<std::pair<float, float>> MiniDrone::logicalCollision()
{
  // Kill the ninja if it touches the mini drone
  auto ninja = sim->getNinja();
  if (ninja->isValidTarget())
  {
    if (Physics::overlapCircleVsCircle(xpos, ypos, RADIUS,
                                       ninja->xpos, ninja->ypos, ninja->RADIUS))
    {
      ninja->kill(0, 0, 0, 0, 0);
    }
  }
  return std::nullopt;
}