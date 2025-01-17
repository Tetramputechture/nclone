#include "one_way_platform.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

OneWayPlatform::OneWayPlatform(Simulation *sim, float xcoord, float ycoord, int orientation)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), orientation(orientation)
{
  auto [dirX, dirY] = Physics::mapOrientationToVector(orientation);
  normalX = dirX;
  normalY = dirY;
}

std::optional<std::tuple<std::pair<float, float>, std::pair<float, float>>>
OneWayPlatform::calculateDepenetration(const Ninja *ninja) const
{
  float dx = ninja->xpos - xpos;
  float dy = ninja->ypos - ypos;
  float lateralDist = dy * normalX - dx * normalY;
  float direction = (ninja->yspeed * normalX - ninja->xspeed * normalY) * lateralDist;

  // The platform has a bigger width if the ninja is moving towards its center
  float radiusScalar = direction < 0 ? 0.91f : 0.51f;
  if (std::abs(lateralDist) < radiusScalar * ninja->RADIUS + SEMI_SIDE)
  {
    float normalDist = dx * normalX + dy * normalY;
    if (0 < normalDist && normalDist <= ninja->RADIUS)
    {
      float normalProj = ninja->xspeed * normalX + ninja->yspeed * normalY;
      if (normalProj <= 0)
      {
        float dxOld = ninja->xposOld - xpos;
        float dyOld = ninja->yposOld - ypos;
        float normalDistOld = dxOld * normalX + dyOld * normalY;
        if (ninja->RADIUS - normalDistOld <= 1.1f)
        {
          return std::make_tuple(
              std::make_pair(normalX, normalY),
              std::make_pair(ninja->RADIUS - normalDist, 0.0f));
        }
      }
    }
  }
  return std::nullopt;
}

void OneWayPlatform::physicalCollision()
{
  auto ninja = sim->getNinja();
  auto depen = calculateDepenetration(ninja);
  if (!depen)
    return;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;
  const auto &[depenLen, _] = penetrations;

  ninja->xpos += depenX * depenLen;
  ninja->ypos += depenY * depenLen;
}

void OneWayPlatform::logicalCollision()
{
  auto ninja = sim->getNinja();
  auto depen = calculateDepenetration(ninja);
  if (!depen)
    return;

  const auto &[normal, _] = *depen;
  const auto &[depenX, depenY] = normal;

  if (std::abs(normalX) == 1)
  {
    ninja->wallNormal = normalX;
    ninja->wallCount++;
  }
}