#include "laser.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"
#include <cmath>

Laser::Laser(Simulation *sim, float xcoord, float ycoord, int orientation, int mode)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord),
      orientation(orientation),
      mode(mode),
      xstart(xcoord),
      ystart(ycoord)
{
  if (mode == 0)
  {
    angle = orientation * M_PI / 4.0f;
    clockwise = orientation >= 4;
  }
  else
  {
    auto [dirX, dirY] = Physics::mapOrientationToVector(orientation);
    xend = xstart + 24.0f * dirX;
    yend = ystart + 24.0f * dirY;
    clockwise = mode >= 3;
  }
}

void Laser::think()
{
  if (mode == 0)
    thinkSpinner();
  else
    thinkSurface();

  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  if (Physics::overlapCircleVsCircle(xpos, ypos, RADIUS, ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    ninja->kill(0, xpos, ypos, 0.0f, 0.0f);
  }
}

void Laser::thinkSpinner()
{
  if (clockwise)
    angle += SPIN_SPEED;
  else
    angle -= SPIN_SPEED;

  if (angle < 0.0f)
    angle += 2.0f * M_PI;
  else if (angle >= 2.0f * M_PI)
    angle -= 2.0f * M_PI;

  xpos = xstart + RADIUS * std::cos(angle);
  ypos = ystart + RADIUS * std::sin(angle);
}

void Laser::thinkSurface()
{
  if (clockwise)
    progress += SURFACE_FLAT_SPEED;
  else
    progress -= SURFACE_FLAT_SPEED;

  if (progress < 0.0f || progress > 1.0f)
  {
    clockwise = !clockwise;
    progress = std::clamp(progress, 0.0f, 1.0f);
  }

  xpos = xstart + (xend - xstart) * progress;
  ypos = ystart + (yend - ystart) * progress;
}

std::vector<float> Laser::getState(bool minimalState) const
{
  auto state = Entity::getState(minimalState);
  if (!minimalState)
  {
    state.push_back(static_cast<float>(orientation));
    state.push_back(static_cast<float>(mode));
    state.push_back(angle);
    state.push_back(progress);
    state.push_back(clockwise ? 1.0f : 0.0f);
  }
  return state;
}