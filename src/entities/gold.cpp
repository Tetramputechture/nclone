#include "gold.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

Gold::Gold(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
}

EntityCollisionResult Gold::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (ninja->getState() == 8) // Don't collect if ninja is in winning state
    return EntityCollisionResult::noCollision();

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    ninja->goldCollected++;
    setActive(false);
    logCollision();
    return EntityCollisionResult::logicalCollision();
  }
  return EntityCollisionResult::noCollision();
}
