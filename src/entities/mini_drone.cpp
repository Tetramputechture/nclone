#include "mini_drone.hpp"

MiniDrone::MiniDrone(Simulation *sim, float xcoord, float ycoord, int orientation, int mode)
    : DroneBase(ENTITY_TYPE, sim, xcoord, ycoord, orientation, mode, DEFAULT_SPEED)
{
}

void MiniDrone::logicalCollision()
{
  // TODO: Implement logical collision
}