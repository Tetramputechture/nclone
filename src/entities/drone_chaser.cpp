#include "drone_chaser.hpp"
#include "../physics/physics.hpp"
#include "../ninja.hpp"

DroneChaser::DroneChaser(Simulation *sim, float xcoord, float ycoord, int orientation, int mode)
    : DroneZap(sim, xcoord, ycoord, orientation, mode)
{
}

void DroneChaser::think()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
    return;

  float dx = ninja->xpos - xpos;
  float dy = ninja->ypos - ypos;
  float dist = std::sqrt(dx * dx + dy * dy);

  if (dist > 0)
  {
    dx /= dist;
    dy /= dist;
    orientation = Physics::mapVectorToOrientation(dx, dy);
  }
}

bool DroneChaser::chooseNextDirectionAndGoal()
{
  auto ninja = sim->getNinja();
  if (!ninja || !ninja->isValidTarget())
  {
    return DroneZap::chooseNextDirectionAndGoal();
  }

  float dx = ninja->xpos - xpos;
  float dy = ninja->ypos - ypos;
  float dist = std::sqrt(dx * dx + dy * dy);

  if (dist > 0)
  {
    dx /= dist;
    dy /= dist;
    orientation = Physics::mapVectorToOrientation(dx, dy);
    auto [dirX, dirY] = DIR_TO_VEC.at(orientation);
    goalX = xpos + GRID_WIDTH * dirX;
    goalY = ypos + GRID_WIDTH * dirY;
  }
  return true;
}