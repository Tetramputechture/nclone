#include "death_ball.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

DeathBall::DeathBall(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
}

void DeathBall::think()
{
  // TODO: Implement think logic
}

EntityCollisionResult DeathBall::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return EntityCollisionResult::noCollision();

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    ninja->kill(0, xpos, ypos, 0.0f, 0.0f);
    return EntityCollisionResult::logicalCollision();
  }
  return EntityCollisionResult::noCollision();
}