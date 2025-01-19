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

void BounceBlock::physicalCollision()
{
  auto ninja = sim->getNinja();
  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + ninja->RADIUS);
  if (!depen)
    return;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;
  const auto &[depenLen, _] = penetrations;

  // Apply 80% of depenetration to block, 20% to ninja
  xpos -= depenX * depenLen * (1.0f - STRENGTH);
  ypos -= depenY * depenLen * (1.0f - STRENGTH);
  xspeed -= depenX * depenLen * (1.0f - STRENGTH);
  yspeed -= depenY * depenLen * (1.0f - STRENGTH);

  ninja->xpos += depenX * depenLen * STRENGTH;
  ninja->ypos += depenY * depenLen * STRENGTH;
  ninja->xspeed += depenX * depenLen * STRENGTH;
  ninja->yspeed += depenY * depenLen * STRENGTH;
}

void BounceBlock::logicalCollision()
{
  auto ninja = sim->getNinja();
  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + ninja->RADIUS + 0.1f);
  if (!depen)
    return;

  const auto &[normal, _] = *depen;
  const auto &[depenX, depenY] = normal;

  if (std::abs(depenX) > std::abs(depenY))
  {
    ninja->wallNormal = depenX > 0 ? 1 : -1;
  }
}