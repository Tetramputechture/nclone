#include "toggle_mine.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

ToggleMine::ToggleMine(Simulation *sim, float xcoord, float ycoord, int state)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord)
{
  setState(state);
}

void ToggleMine::think()
{
  auto ninja = sim->getNinja();
  if (!ninja->isValidTarget())
  {
    if (state == 2)
    {
      setState(1);
    }
    return;
  }

  if (state == 1)
  { // untoggled state
    if (Physics::overlapCircleVsCircle(
            xpos, ypos, RADII[state],
            ninja->xpos, ninja->ypos, ninja->RADIUS))
    {
      setState(2); // set to toggling state
    }
  }
  else if (state == 2)
  { // toggling state
    if (!Physics::overlapCircleVsCircle(
            xpos, ypos, RADII[state],
            ninja->xpos, ninja->ypos, ninja->RADIUS))
    {
      setState(0); // set to toggled state
    }
  }
}

std::optional<EntityCollisionResult> ToggleMine::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (!ninja->isValidTarget() || state == 2)
    return std::nullopt;

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADII[state],
          ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    if (state == 0)
    { // toggled state
      setState(1);
      ninja->kill(0, 0, 0, 0, 0);
    }
  }
  return std::nullopt;
}

void ToggleMine::setState(int newState)
{
  if (newState >= 0 && newState <= 2)
  {
    state = newState;
    logCollision(state);
  }
}

std::vector<float> ToggleMine::getState(bool minimalState) const
{
  auto baseState = Entity::getState(minimalState);
  return baseState;
}