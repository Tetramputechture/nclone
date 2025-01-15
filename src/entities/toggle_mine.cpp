#include "toggle_mine.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

EntityToggleMine::EntityToggleMine(int entityType, Simulation *sim, float xcoord, float ycoord, int state)
    : Entity(entityType, sim, xcoord, ycoord), state(state)
{
}

void EntityToggleMine::think()
{
  if (state == 2)
  { // toggling state
    toggleTimer++;
    if (toggleTimer >= 30)
    {
      setState(0); // set to toggled state
    }
  }
}

void EntityToggleMine::logicalCollision()
{
  if (!active || state == 2)
    return;

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADII[state],
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    if (state == 0)
    { // toggled state
      sim->getNinja()->kill(0, xpos, ypos, 0, 0);
    }
    else
    {              // untoggled state
      setState(2); // set to toggling state
    }
  }
}

void EntityToggleMine::setState(int newState)
{
  state = newState;
  toggleTimer = 0;
}

std::vector<float> EntityToggleMine::getState(bool minimalState) const
{
  auto baseState = Entity::getState(minimalState);
  if (!minimalState)
  {
    baseState.push_back(static_cast<float>(state));
    baseState.push_back(static_cast<float>(toggleTimer));
  }
  return baseState;
}