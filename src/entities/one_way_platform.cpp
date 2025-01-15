#include "one_way_platform.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

OneWayPlatform::OneWayPlatform(Simulation *sim, float xcoord, float ycoord, int orientation)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), orientation(orientation)
{
}

std::optional<std::tuple<std::pair<float, float>, std::pair<float, float>>>
OneWayPlatform::calculateDepenetration(const Ninja *ninja) const
{
  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
  if (!depen)
    return std::nullopt;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;
  const auto &[depenLen, _] = penetrations;

  auto [dirX, dirY] = Physics::mapOrientationToVector(orientation);

  // Only depenetrate if moving towards the platform from the correct side
  if (depenX * dirX + depenY * dirY >= 0)
    return std::nullopt;

  return depen;
}

void OneWayPlatform::physicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

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
  if (!ninja || !ninja->isValidTarget())
    return;

  auto depen = calculateDepenetration(ninja);
  if (!depen)
    return;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;

  if (depenY < -0.0001f)
  {
    ninja->floorCount++;
    ninja->floorNormalX += depenX;
    ninja->floorNormalY += depenY;
  }
  else
  {
    ninja->ceilingCount++;
    ninja->ceilingNormalX += depenX;
    ninja->ceilingNormalY += depenY;
  }
}