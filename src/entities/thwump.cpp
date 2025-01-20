#include "thwump.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

Thwump::Thwump(Simulation *sim, float xcoord, float ycoord, int orientation)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), orientation(orientation),
      xstart(xcoord), ystart(ycoord)
{
  logPositions = sim->getConfig().logData;
  isHorizontal = orientation % 4 == 0;
  std::tie(dirX, dirY) = Physics::mapOrientationToVector(orientation);
}

void Thwump::think()
{
  if (state == 0)
  {
    auto ninja = sim->getNinja();
    if (!ninja || !ninja->isValidTarget())
      return;

    // Check if ninja is in activation range
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
    // Check for wall collision in forward direction
    if (isHorizontal)
    {
      int cellX = std::floor(xpos / 12);
      int cellXNew = std::floor((xpos + dirX * 11) / 12);
      if (cellX != cellXNew)
      {
        int cellY1 = std::floor((ypos - 11) / 12);
        int cellY2 = std::floor((ypos + 11) / 12);
        if (!Physics::isEmptyColumn(*sim, cellX, cellY1, cellY2, dirX > 0 ? 1 : -1))
        {
          setState(2);
          return;
        }
      }
    }
    else
    {
      int cellY = std::floor(ypos / 12);
      int cellYNew = std::floor((ypos + dirY * 11) / 12);
      if (cellY != cellYNew)
      {
        int cellX1 = std::floor((xpos - 11) / 12);
        int cellX2 = std::floor((xpos + 11) / 12);
        if (!Physics::isEmptyRow(*sim, cellX1, cellX2, cellY, dirY > 0 ? 1 : -1))
        {
          setState(2);
          return;
        }
      }
    }

    // Check if we've gone too far from start
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
    // Check if we're close enough to starting position to reset
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
  xposOld = xpos;
  yposOld = ypos;

  if (state == 1)
  {
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

std::optional<EntityCollisionResult> Thwump::physicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + ninja->RADIUS);
  if (!depen)
    return std::nullopt;

  const auto &[normal, penetrations] = *depen;
  return EntityCollisionResult(normal.first, normal.second, penetrations.first, penetrations.second);
}

std::optional<EntityCollisionResult> Thwump::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  auto depen = Physics::penetrationSquareVsPoint(xpos, ypos, ninja->xpos, ninja->ypos, SEMI_SIDE + 0.1f);
  if (!depen)
    return std::nullopt;

  const auto &[normal, penetrations] = *depen;

  // Check for lethal collision with charging face
  if (isHorizontal)
  {
    float dx = (SEMI_SIDE + 2) * dirX;
    float dy = SEMI_SIDE - 2;
    float px1 = xpos + dx;
    float py1 = ypos - dy;
    float px2 = xpos + dx;
    float py2 = ypos + dy;

    if (Physics::overlapCircleVsSegment(ninja->xpos, ninja->ypos, ninja->RADIUS + 2, px1, py1, px2, py2))
    {
      ninja->kill(0, 0, 0, 0, 0);
    }
  }
  else
  {
    float dx = SEMI_SIDE - 2;
    float dy = (SEMI_SIDE + 2) * dirY;
    float px1 = xpos - dx;
    float py1 = ypos + dy;
    float px2 = xpos + dx;
    float py2 = ypos + dy;

    if (Physics::overlapCircleVsSegment(ninja->xpos, ninja->ypos, ninja->RADIUS + 2, px1, py1, px2, py2))
    {
      ninja->kill(0, 0, 0, 0, 0);
    }
  }

  return EntityCollisionResult(normal.first, normal.second, penetrations.first, penetrations.second);
}

void Thwump::setState(int newState)
{
  if (newState >= 0 && newState <= 2)
  {
    state = newState;
    logCollision(state);
  }
}