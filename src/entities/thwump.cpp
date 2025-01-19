#include "thwump.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

Thwump::Thwump(Simulation *sim, float xcoord, float ycoord, int orientation)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), orientation(orientation),
      xstart(xcoord), ystart(ycoord)
{
}

void Thwump::think()
{
  if (!active)
    return;

  if (state == 0)
  {
    auto ninja = sim->getNinja();
    if (!ninja || !ninja->isValidTarget())
      return;

    auto [dirX, dirY] = Physics::mapOrientationToVector(orientation);
    float dx = ninja->xpos - xpos;
    float dy = ninja->ypos - ypos;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist > 0)
    {
      dx /= dist;
      dy /= dist;
      if (dx * dirX + dy * dirY > 0.9f)
      {
        // Check if there's a clear line of sight to the ninja
        if (!Physics::raycastVsPlayer(*sim, xpos, ypos, ninja->xpos, ninja->ypos, ninja->RADIUS))
        {
          setState(1);
        }
      }
    }
  }
  else if (state == 1)
  {
    auto [dirX, dirY] = Physics::mapOrientationToVector(orientation);
    float dx = xpos - xstart;
    float dy = ypos - ystart;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist > 0)
    {
      dx /= dist;
      dy /= dist;
      if (dx * dirX + dy * dirY > 0.9f)
      {
        setState(2);
      }
    }
  }
  else if (state == 2)
  {
    float dx = xpos - xstart;
    float dy = ypos - ystart;
    if (dx * dx + dy * dy < 1.0f)
    {
      xpos = xstart;
      ypos = ystart;
      setState(0);
    }
  }
}

void Thwump::move()
{
  if (!active)
    return;

  xposOld = xpos;
  yposOld = ypos;

  if (state == 1)
  {
    auto [dirX, dirY] = Physics::mapOrientationToVector(orientation);
    xpos += dirX * FORWARD_SPEED;
    ypos += dirY * FORWARD_SPEED;
  }
  else if (state == 2)
  {
    float dx = xstart - xpos;
    float dy = ystart - ypos;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > 0)
    {
      dx /= dist;
      dy /= dist;
      xpos += dx * BACKWARD_SPEED;
      ypos += dy * BACKWARD_SPEED;
    }
  }

  gridMove();
}

EntityCollisionResult Thwump::physicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return EntityCollisionResult::noCollision();

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
  if (!depen)
    return EntityCollisionResult::noCollision();

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;
  const auto &[depenLen, _] = penetrations;

  ninja->xpos += depenX * depenLen;
  ninja->ypos += depenY * depenLen;
  ninja->xspeed += depenX * depenLen;
  ninja->yspeed += depenY * depenLen;
  return EntityCollisionResult::physicalCollision(depenX, depenY, depenX, depenY);
}

EntityCollisionResult Thwump::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return EntityCollisionResult::noCollision();

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
  if (!depen)
    return EntityCollisionResult::noCollision();

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;

  if (state == 1)
  {
    ninja->kill(0, xpos, ypos, 0.0f, 0.0f);
    return EntityCollisionResult::logicalCollision();
  }

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
  return EntityCollisionResult::physicalCollision(depenX, depenY, depenX, depenY);
}

void Thwump::setState(int newState)
{
  if (newState >= 0 && newState <= 2)
  {
    state = newState;
    logCollision(state);
  }
}