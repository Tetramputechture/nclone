#include "exit.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

EntityExit::EntityExit(int entityType, Simulation *sim, float xcoord, float ycoord)
    : Entity(entityType, sim, xcoord, ycoord)
{
}

void EntityExit::logicalCollision()
{
  if (!active)
    return;

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    sim->getNinja()->win();
    logCollision();
  }
}