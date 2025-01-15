#include "gold.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

EntityGold::EntityGold(int entityType, Simulation *sim, float xcoord, float ycoord)
    : Entity(entityType, sim, xcoord, ycoord)
{
}

void EntityGold::logicalCollision()
{
  if (!active)
    return;

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    active = false;
    sim->getNinja()->goldCollected++;
    logCollision();
  }
}