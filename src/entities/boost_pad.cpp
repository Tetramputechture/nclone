#include "boost_pad.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

BoostPad::BoostPad(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
  isTouchingNinja = false;
}

void BoostPad::move()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
  {
    isTouchingNinja = false;
    return;
  }

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    if (!isTouchingNinja)
    {
      float velNorm = std::sqrt(ninja->xspeed * ninja->xspeed + ninja->yspeed * ninja->yspeed);
      if (velNorm > 0)
      {
        float xBoost = 2.0f * ninja->xspeed / velNorm;
        float yBoost = 2.0f * ninja->yspeed / velNorm;
        ninja->xspeed += xBoost;
        ninja->yspeed += yBoost;
      }
      isTouchingNinja = true;
    }
  }
  else
  {
    isTouchingNinja = false;
  }
}