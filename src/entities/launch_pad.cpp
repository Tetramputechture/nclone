#include "launch_pad.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

LaunchPad::LaunchPad(Simulation *sim, float xcoord, float ycoord, int orientation)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), orientation(orientation)
{
}

std::optional<EntityCollisionResult> LaunchPad::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  if (Physics::overlapCircleVsCircle(xpos, ypos, RADIUS, ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    auto [normalX, normalY] = Physics::mapOrientationToVector(orientation);

    if ((xpos - (ninja->xpos - ninja->RADIUS * normalX)) * normalX +
            (ypos - (ninja->ypos - ninja->RADIUS * normalY)) * normalY >=
        -0.1f)
    {
      float yboostScale = 1.0f;
      if (normalY < 0)
      {
        yboostScale = 1.0f - normalY;
      }

      float xboost = normalX * BOOST;
      float yboost = normalY * BOOST * yboostScale;

      return EntityCollisionResult{xboost, yboost};
    }
  }
  return std::nullopt;
}
