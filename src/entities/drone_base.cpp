#include "drone_base.hpp"
#include "../physics/physics.hpp"

// Initialize static members
const std::unordered_map<int, std::array<float, 2>> DroneBase::DIR_TO_VEC = {
    {0, {1.0f, 0.0f}},
    {1, {0.0f, 1.0f}},
    {2, {-1.0f, 0.0f}},
    {3, {0.0f, -1.0f}}};

// Dictionary to choose the next direction from the patrolling mode of the drone.
// Patrolling modes : {0:follow wall CW, 1:follow wall CCW, 2:wander CW, 3:wander CCW}
// Directions : {0:keep forward, 1:turn right, 2:go backward, 3:turn left}
const std::unordered_map<int, std::array<int, 4>> DroneBase::DIR_LIST = {
    {0, {1, 0, 3, 2}},
    {1, {3, 0, 1, 2}},
    {2, {0, 1, 3, 2}},
    {3, {0, 3, 1, 2}}};

DroneBase::DroneBase(int entityType, Simulation *sim, float xcoord, float ycoord, int orientation, int mode, float speed)
    : Entity(entityType, sim, xcoord, ycoord),
      orientation(orientation),
      mode(mode),
      speed(speed),
      goalX(xcoord),
      goalY(ycoord),
      xpos2(xcoord),
      ypos2(ycoord)
{
  turn(orientation / 2);
}

void DroneBase::turn(int newDir)
{
  dirOld = dir;
  dir = newDir;
  if (sim->getConfig().fullExport)
  {
    logCollision(dir);
  }
}

void DroneBase::move()
{
  auto [dirX, dirY] = DIR_TO_VEC.at(dir);
  float xspeed = speed * dirX;
  float yspeed = speed * dirY;
  float dx = goalX - xpos;
  float dy = goalY - ypos;
  float dist = std::sqrt(dx * dx + dy * dy);

  // If the drone has reached or passed the center of the cell, choose the next cell to go to
  if (dist < 0.000001f || (dx * (goalX - (xpos + xspeed)) + dy * (goalY - (ypos + yspeed))) < 0)
  {
    xpos = goalX;
    ypos = goalY;
    bool canMove = chooseNextDirectionAndGoal();
    if (canMove)
    {
      float disp = speed - dist;
      auto [newDirX, newDirY] = DIR_TO_VEC.at(dir);
      xpos += disp * newDirX;
      ypos += disp * newDirY;
    }
  }
  else
  {
    xpos += xspeed;
    ypos += yspeed;
  }
  gridMove();
}

bool DroneBase::testNextDirectionAndGoal(int dir)
{
  auto [dirX, dirY] = DIR_TO_VEC.at(dir);
  float nextGoalX = xpos + GRID_WIDTH * dirX;
  float nextGoalY = ypos + GRID_WIDTH * dirY;

  if (dirY == 0)
  {
    int cellX = static_cast<int>(std::floor((xpos + dirX * RADIUS) / 12.0f));
    int cellXTarget = static_cast<int>(std::floor((nextGoalX + dirX * RADIUS) / 12.0f));
    int cellY1 = static_cast<int>(std::floor((ypos - RADIUS) / 12.0f));
    int cellY2 = static_cast<int>(std::floor((ypos + RADIUS) / 12.0f));

    while (cellX != cellXTarget)
    {
      if (!Physics::isEmptyColumn(*sim, cellX, cellY1, cellY2, dirX))
      {
        return false;
      }
      cellX += dirX;
    }
  }
  else
  {
    int cellY = static_cast<int>(std::floor((ypos + dirY * RADIUS) / 12.0f));
    int cellYTarget = static_cast<int>(std::floor((nextGoalY + dirY * RADIUS) / 12.0f));
    int cellX1 = static_cast<int>(std::floor((xpos - RADIUS) / 12.0f));
    int cellX2 = static_cast<int>(std::floor((xpos + RADIUS) / 12.0f));

    while (cellY != cellYTarget)
    {
      if (!Physics::isEmptyRow(*sim, cellX1, cellX2, cellY, dirY))
      {
        return false;
      }
      cellY += dirY;
    }
  }

  goalX = nextGoalX;
  goalY = nextGoalY;
  return true;
}

bool DroneBase::chooseNextDirectionAndGoal()
{
  const auto &dirList = DIR_LIST.at(mode);
  for (int i = 0; i < 4; i++)
  {
    int newDir = (dir + dirList[i]) % 4;
    if (testNextDirectionAndGoal(newDir))
    {
      turn(newDir);
      return true;
    }
  }
  return false;
}

std::vector<float> DroneBase::getState(bool minimalState) const
{
  auto state = Entity::getState(minimalState);
  if (!minimalState)
  {
    state.push_back(static_cast<float>(mode));
    state.push_back(dir == -1 ? 0.5f : (static_cast<float>(dir) + 1.0f) / 2.0f);
    state.push_back(static_cast<float>(orientation) / 7.0f);
  }
  return state;
}