#include "thwump.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

Thwump::Thwump(Simulation *sim, float xcoord, float ycoord, int orientation)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord),
      orientation(orientation),
      xstart(xcoord),
      ystart(ycoord)
{
}

void Thwump::setState(int newState)
{
  state = newState;
  if (state == 1)
    speed = FORWARD_SPEED;
  else if (state == 2)
    speed = BACKWARD_SPEED;
  else
    speed = 0.0f;
}

void Thwump::move()
{
  xposOld = xpos;
  yposOld = ypos;

  if (state != 0)
  {
    auto [dirX, dirY] = Physics::mapOrientationToVector(orientation);
    if (state == 2)
    {
      dirX = -dirX;
      dirY = -dirY;
    }

    xpos += speed * dirX;
    ypos += speed * dirY;
  }
}

void Thwump::think()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  if (state == 0)
  {
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

void Thwump::physicalCollision()
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

  ninja->xpos += depenX * depenLen;
  ninja->ypos += depenY * depenLen;
  ninja->xspeed += depenX * depenLen;
  ninja->yspeed += depenY * depenLen;
}

void Thwump::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
  if (!depen)
    return;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;

  if (state == 1)
  {
    ninja->kill(0, xpos, ypos, 0.0f, 0.0f);
    return;
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
}