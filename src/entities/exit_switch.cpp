#include "exit_switch.hpp"
#include "exit.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"

EntityExitSwitch::EntityExitSwitch(int entityType, Simulation *sim, float xcoord, float ycoord, EntityExit *parent)
    : Entity(entityType, sim, xcoord, ycoord), parent(parent)
{
}

void EntityExitSwitch::logicalCollision()
{
  if (!active)
    return;

  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          sim->getNinja()->xpos, sim->getNinja()->ypos, sim->getNinja()->RADIUS))
  {
    active = false;
    parent->active = true;
    logCollision();
  }
}