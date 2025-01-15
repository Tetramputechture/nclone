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
    toggle_timer++;
    if (toggle_timer >= 30)
    {
      set_state(0); // set to toggled state
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
    {               // untoggled state
      set_state(2); // set to toggling state
    }
  }
}

void EntityToggleMine::set_state(int new_state)
{
  state = new_state;
  toggle_timer = 0;
}

std::vector<float> EntityToggleMine::getState(bool minimal_state) const
{
  auto base_state = Entity::getState(minimal_state);
  if (!minimal_state)
  {
    base_state.push_back(static_cast<float>(state));
    base_state.push_back(static_cast<float>(toggle_timer));
  }
  return base_state;
}