#include "bounce_block.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

BounceBlock::BounceBlock(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
}

void BounceBlock::move()
{
  xspeedOld = xspeed;
  yspeedOld = yspeed;
  xposOld = xpos;
  yposOld = ypos;
  xpos += xspeed;
  ypos += yspeed;
}

void BounceBlock::physicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
  if (!depen)
    return;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;
  const auto &[depenLen, _] = penetrations;

  // Apply depenetration to ninja
  ninja->xpos += depenX * depenLen;
  ninja->ypos += depenY * depenLen;

  // Calculate relative velocity
  float relVelX = ninja->xspeed - xspeed;
  float relVelY = ninja->yspeed - yspeed;

  // Apply impulse
  float impulse = -(relVelX * depenX + relVelY * depenY) * (1.0f + STRENGTH);
  float impulseX = impulse * depenX;
  float impulseY = impulse * depenY;

  ninja->xspeed += impulseX;
  ninja->yspeed += impulseY;
  xspeed -= impulseX;
  yspeed -= impulseY;
}

void BounceBlock::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
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