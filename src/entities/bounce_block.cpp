#include "bounce_block.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

BounceBlock::BounceBlock(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
  logPositions = sim->getConfig().fullExport;
  xorigin = xcoord;
  yorigin = ycoord;
}

void BounceBlock::move()
{
  // Apply dampening
  xspeed *= DAMPENING;
  yspeed *= DAMPENING;

  // Update position
  xpos += xspeed;
  ypos += yspeed;

  // Apply spring force
  float xforce = STIFFNESS * (xorigin - xpos);
  float yforce = STIFFNESS * (yorigin - ypos);
  xpos += xforce;
  ypos += yforce;
  xspeed += xforce;
  yspeed += yforce;

  gridMove();
}

std::optional<EntityCollisionResult> BounceBlock::physicalCollision()
{
  auto ninja = sim->getNinja();
  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + ninja->RADIUS);
  if (!depen)
    return std::nullopt;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;
  const auto &[depenLen, depenLen2] = penetrations;

  // Apply 80% of depenetration to block, 20% to ninja
  xpos -= depenX * depenLen * (1.0f - STRENGTH);
  ypos -= depenY * depenLen * (1.0f - STRENGTH);
  xspeed -= depenX * depenLen * (1.0f - STRENGTH);
  yspeed -= depenY * depenLen * (1.0f - STRENGTH);

  return EntityCollisionResult(
      depenX,
      depenY,
      depenLen * STRENGTH,
      depenLen2);
}

std::optional<EntityCollisionResult> BounceBlock::logicalCollision()
{
  auto ninja = sim->getNinja();
  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + ninja->RADIUS + 0.1f);
  if (!depen)
    return;

  const auto &[normal, _] = *depen;
  const auto &[depenX, depenY] = normal;

  return EntityCollisionResult(depenX);
}