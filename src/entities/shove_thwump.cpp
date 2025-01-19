#include "shove_thwump.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

ShoveThwump::ShoveThwump(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), xorigin(xcoord), yorigin(ycoord)
{
}

void ShoveThwump::setState(int newState)
{
  state = newState;
}

bool ShoveThwump::moveIfPossible(float xdir, float ydir, float speed)
{
  if (ydir == 0.0f)
  {
    float xposNew = xpos + xdir * speed;
    int cellX = static_cast<int>(xpos / 12.0f);
    int cellXNew = static_cast<int>(xposNew / 12.0f);

    if (cellX != cellXNew)
    {
      int cellY1 = static_cast<int>((ypos - 8.0f) / 12.0f);
      int cellY2 = static_cast<int>((ypos + 8.0f) / 12.0f);

      if (!Physics::isEmptyColumn(*sim, cellX, cellY1, cellY2, xdir > 0.0f ? 1 : -1))
      {
        setState(3);
        return false;
      }
    }
    xpos = xposNew;
  }
  else
  {
    float yposNew = ypos + ydir * speed;
    int cellY = static_cast<int>(ypos / 12.0f);
    int cellYNew = static_cast<int>(yposNew / 12.0f);

    if (cellY != cellYNew)
    {
      int cellX1 = static_cast<int>((xpos - 8.0f) / 12.0f);
      int cellX2 = static_cast<int>((xpos + 8.0f) / 12.0f);

      if (!Physics::isEmptyRow(*sim, cellX1, cellX2, cellY, ydir > 0.0f ? 1 : -1))
      {
        setState(3);
        return false;
      }
    }
    ypos = yposNew;
  }

  gridMove();
  return true;
}

void ShoveThwump::think()
{
  if (state == 1)
  {
    if (activated)
    {
      activated = false;
      return;
    }
    setState(2);
  }
  else if (state == 3)
  {
    float originDist = std::abs(xpos - xorigin) + std::abs(ypos - yorigin);
    if (originDist >= 1.0f)
    {
      moveIfPossible(xdir, ydir, 1.0f);
    }
    else
    {
      xpos = xorigin;
      ypos = yorigin;
      setState(0);
    }
  }
  else if (state == 2)
  {
    moveIfPossible(-xdir, -ydir, 4.0f);
  }
}

void ShoveThwump::move()
{
  // Movement is handled in think() and moveIfPossible()
  xposOld = xpos;
  yposOld = ypos;
}

std::optional<EntityCollisionResult> ShoveThwump::physicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  if (state <= 1)
  {
    auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + ninja->RADIUS);
    if (!depen)
      return std::nullopt;

    const auto &[normal, penetrations] = *depen;
    const auto &[depenX, depenY] = normal;

    if (state == 0 || xdir * depenX + ydir * depenY >= 0.01f)
    {
      return EntityCollisionResult(depenX, depenY, penetrations.first, penetrations.second);
    }
  }

  return std::nullopt;
}

std::optional<EntityCollisionResult> ShoveThwump::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + ninja->RADIUS + 0.1f);
  if (depen && state <= 1)
  {
    const auto &[normal, penetrations] = *depen;
    const auto &[depenX, depenY] = normal;

    if (state == 0)
    {
      activated = true;
      if (penetrations.second > 0.2f)
      {
        xdir = depenX;
        ydir = depenY;
        setState(1);
      }
    }
    else if (state == 1)
    {
      if (xdir * depenX + ydir * depenY >= 0.01f)
      {
        activated = true;
      }
      else
      {
        return std::nullopt;
      }
    }
    return EntityCollisionResult(depenX, 0.0f, 0.0f, 0.0f);
  }

  if (Physics::overlapCircleVsCircle(ninja->xpos, ninja->ypos, ninja->RADIUS, xpos, ypos, RADIUS))
  {
    ninja->kill(0, 0.0f, 0.0f, 0.0f, 0.0f);
  }

  return std::nullopt;
}
