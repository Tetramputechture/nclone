#include "death_ball.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"
#include <cmath>

DeathBall::DeathBall(Simulation *sim, float xcoord, float ycoord)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
}

void DeathBall::think()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
  {
    // If no valid targets, decelerate ball to a stop
    xspeed *= DRAG_NO_TARGET;
    yspeed *= DRAG_NO_TARGET;
  }
  else
  {
    // Apply acceleration towards closest ninja
    float dx = ninja->getXPos() - xpos;
    float dy = ninja->getYPos() - ypos;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > 0)
    {
      dx /= dist;
      dy /= dist;
    }
    xspeed += dx * ACCELERATION;
    yspeed += dy * ACCELERATION;

    // Apply drag if speed exceeds MAX_SPEED
    float speed = std::sqrt(xspeed * xspeed + yspeed * yspeed);
    if (speed > MAX_SPEED)
    {
      float newSpeed = (speed - MAX_SPEED) * DRAG_MAX_SPEED;
      if (newSpeed <= 0.01f)
      {
        newSpeed = 0;
      }
      newSpeed += MAX_SPEED;
      xspeed = xspeed / speed * newSpeed;
      yspeed = yspeed / speed * newSpeed;
    }
  }

  float xposOld = xpos;
  float yposOld = ypos;
  xpos += xspeed;
  ypos += yspeed;

  // Interpolation routine for high-speed wall collisions
  float time = Physics::sweepCircleVsTiles(*sim, xposOld, yposOld, xspeed, yspeed, RADIUS2 * 0.5f);
  xpos = xposOld + time * xspeed;
  ypos = yposOld + time * yspeed;

  // Depenetration routine for collision against tiles
  float xnormal = 0, ynormal = 0;
  for (int i = 0; i < 16; i++)
  {
    auto result = Physics::getSingleClosestPoint(*sim, xpos, ypos, RADIUS2);
    if (!result)
      break;

    auto [isBackFacing, closestPoint] = *result;
    float dx = xpos - closestPoint.first;
    float dy = ypos - closestPoint.second;
    float dist = std::sqrt(dx * dx + dy * dy);
    float depenLen = RADIUS2 - dist * (isBackFacing ? -1.0f : 1.0f);

    if (depenLen < 0.0000001f)
      break;

    if (dist == 0)
      return;

    float xnorm = dx / dist;
    float ynorm = dy / dist;
    xpos += xnorm * depenLen;
    ypos += ynorm * depenLen;
    xnormal += xnorm;
    ynormal += ynorm;
  }

  // If there has been tile collision, project speed onto surface and add bounce if applicable
  float normalLen = std::sqrt(xnormal * xnormal + ynormal * ynormal);
  if (normalLen > 0)
  {
    float dx = xnormal / normalLen;
    float dy = ynormal / normalLen;
    float dotProduct = xspeed * dx + yspeed * dy;
    if (dotProduct < 0)
    {
      float speed = std::sqrt(xspeed * xspeed + yspeed * yspeed);
      float bounceStrength = speed <= 1.35f ? 1.0f : 2.0f;
      xspeed -= dx * dotProduct * bounceStrength;
      yspeed -= dy * dotProduct * bounceStrength;
    }
  }

  // Handle bounces with other death balls
  const auto &deathBalls = sim->getEntitiesOfType(ENTITY_TYPE);

  // Find our index in the death balls list
  size_t myIndex = 0;
  for (; myIndex < deathBalls.size(); myIndex++)
  {
    if (deathBalls[myIndex].get() == this)
    {
      break;
    }
  }

  // Only check collisions with death balls that come after us in the list
  // to avoid processing each collision twice
  for (size_t i = myIndex + 1; i < deathBalls.size(); i++)
  {
    auto otherBall = dynamic_cast<DeathBall *>(deathBalls[i].get());
    if (!otherBall)
      continue;

    float dx = xpos - otherBall->getXPos();
    float dy = ypos - otherBall->getYPos();
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 16)
    {
      dx = dx / dist * 4;
      dy = dy / dist * 4;
      xspeed += dx;
      yspeed += dy;
      otherBall->setXSpeed(otherBall->getXSpeed() - dx);
      otherBall->setYSpeed(otherBall->getYSpeed() - dy);
    }
  }

  gridMove();
}

std::optional<EntityCollisionResult> DeathBall::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return std::nullopt;

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          ninja->getXPos(), ninja->getYPos(), ninja->RADIUS))
  {
    float dx = xpos - ninja->getXPos();
    float dy = ypos - ninja->getYPos();
    float dist = std::sqrt(dx * dx + dy * dy);
    xspeed += dx / dist * 10;
    yspeed += dy / dist * 10;
    ninja->kill(0, 0, 0, 0, 0);
    return std::optional<EntityCollisionResult>(EntityCollisionResult(1.0f));
  }
  return std::nullopt;
}