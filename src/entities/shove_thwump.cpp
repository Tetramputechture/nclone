#include "shove_thwump.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

ShoveThwump::ShoveThwump(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), xstart(xcoord), ystart(ycoord)
{
}

void ShoveThwump::setState(int newState)
{
  state = newState;
  if (state == 1)
    speed = FORWARD_SPEED;
  else if (state == 2)
    speed = BACKWARD_SPEED;
  else
    speed = 0.0f;
}

void ShoveThwump::think()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  if (state == 0)
  {
    float dx = ninja->xpos - xpos;
    float dy = ninja->ypos - ypos;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist > 0)
    {
      dx /= dist;
      dy /= dist;
      // Check if there's a clear line of sight to the ninja
      if (!Physics::raycastVsPlayer(*sim, xpos, ypos, ninja->xpos, ninja->ypos, ninja->RADIUS))
      {
        setState(1);
      }
    }
  }
  else if (state == 1)
  {
    float dx = xpos - xstart;
    float dy = ypos - ystart;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist > 0)
    {
      dx /= dist;
      dy /= dist;
      if (dist > 48.0f)
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

void ShoveThwump::move()
{
  xposOld = xpos;
  yposOld = ypos;

  if (state != 0)
  {
    float dx = 0.0f;
    float dy = 0.0f;

    if (state == 1)
    {
      auto ninja = sim->getNinja();
      if (ninja && ninja->isValidTarget())
      {
        dx = ninja->xpos - xpos;
        dy = ninja->ypos - ypos;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist > 0)
        {
          dx /= dist;
          dy /= dist;
        }
      }
    }
    else if (state == 2)
    {
      dx = xstart - xpos;
      dy = ystart - ypos;
      float dist = std::sqrt(dx * dx + dy * dy);
      if (dist > 0)
      {
        dx /= dist;
        dy /= dist;
      }
    }

    xpos += speed * dx;
    ypos += speed * dy;
  }

  gridMove();
}

std::optional<std::pair<float, float>> ShoveThwump::physicalCollision()
{
  if (state > 1)
    return std::nullopt;

  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
  if (!depen)
    return std::nullopt;

  const auto &[normal, penetrations] = *depen;
  const auto &[depenX, depenY] = normal;
  const auto &[depenLen, _] = penetrations;

  ninja->xpos += depenX * depenLen;
  ninja->ypos += depenY * depenLen;
  ninja->xspeed += depenX * depenLen;
  ninja->yspeed += depenY * depenLen;
  return std::make_pair(depenX, depenY);
}

std::optional<std::pair<float, float>> ShoveThwump::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE);
  if (!depen)
    return std::nullopt;

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