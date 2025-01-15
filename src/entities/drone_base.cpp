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
    {2, {1, 0, 3, 2}},
    {3, {3, 0, 1, 2}}};

DroneBase::DroneBase(int entityType, Simulation *sim, float xcoord, float ycoord, int orientation, int mode, float speed)
    : Entity(entityType, sim, xcoord, ycoord),
      orientation(orientation),
      mode(mode),
      speed(speed)
{
  auto [dirX, dirY] = DIR_TO_VEC.at(orientation);
  goalX = xpos + GRID_WIDTH * dirX;
  goalY = ypos + GRID_WIDTH * dirY;
}

void DroneBase::turn(int dir)
{
  orientation = (orientation + dir) % 4;
  if (orientation < 0)
    orientation += 4;

  auto [dirX, dirY] = DIR_TO_VEC.at(orientation);
  goalX = xpos + GRID_WIDTH * dirX;
  goalY = ypos + GRID_WIDTH * dirY;
}

void DroneBase::move()
{
  float dx = goalX - xpos;
  float dy = goalY - ypos;
  float dist = std::sqrt(dx * dx + dy * dy);

  if (dist <= speed)
  {
    xpos = goalX;
    ypos = goalY;
    chooseNextDirectionAndGoal();
  }
  else
  {
    xpos += speed * dx / dist;
    ypos += speed * dy / dist;
  }
}

bool DroneBase::testNextDirectionAndGoal(int dir)
{
  int nextOrientation = (orientation + dir) % 4;
  if (nextOrientation < 0)
    nextOrientation += 4;

  auto [dirX, dirY] = DIR_TO_VEC.at(nextOrientation);
  float nextGoalX = xpos + GRID_WIDTH * dirX;
  float nextGoalY = ypos + GRID_WIDTH * dirY;

  int xcoord1 = static_cast<int>(std::floor(xpos / GRID_WIDTH));
  int ycoord1 = static_cast<int>(std::floor(ypos / GRID_WIDTH));
  int xcoord2 = static_cast<int>(std::floor(nextGoalX / GRID_WIDTH));
  int ycoord2 = static_cast<int>(std::floor(nextGoalY / GRID_WIDTH));

  if (dirX != 0)
  {
    return Physics::isEmptyColumn(*sim, dirX > 0 ? xcoord1 + 1 : xcoord1,
                                  std::min(ycoord1, ycoord2),
                                  std::max(ycoord1, ycoord2),
                                  dirX > 0 ? 1 : -1);
  }
  else
  {
    return Physics::isEmptyRow(*sim, std::min(xcoord1, xcoord2),
                               std::max(xcoord1, xcoord2),
                               dirY > 0 ? ycoord1 + 1 : ycoord1,
                               dirY > 0 ? 1 : -1);
  }
}

void DroneBase::chooseNextDirectionAndGoal()
{
  const auto &dirList = DIR_LIST.at(mode);
  for (int dir : dirList)
  {
    if (testNextDirectionAndGoal(dir))
    {
      turn(dir);
      return;
    }
  }
}

std::vector<float> DroneBase::getState(bool minimalState) const
{
  auto state = Entity::getState(minimalState);
  if (!minimalState)
  {
    state.push_back(static_cast<float>(orientation));
    state.push_back(static_cast<float>(mode));
  }
  return state;
}