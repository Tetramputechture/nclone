#include "exit_switch.hpp"
#include "exit_door.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../physics/physics.hpp"

ExitSwitch::ExitSwitch(Simulation *sim, float xcoord, float ycoord, ExitDoor *parent)
    : Entity(ENTITY_TYPE, sim, xcoord, ycoord), parent(parent)
{
}

std::optional<EntityCollisionResult> ExitSwitch::logicalCollision()
{
  auto ninja = sim->getNinja();
  if (Physics::overlapCircleVsCircle(
          xpos, ypos, RADIUS,
          ninja->xpos, ninja->ypos, ninja->RADIUS))
  {
    setActive(false);
    sim->addEntity(std::shared_ptr<Entity>(parent));
    logCollision();
  }
  return std::nullopt;
}