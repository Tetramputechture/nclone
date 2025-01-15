#include "entity.hpp"
#include "simulation.hpp"
#include "sim_config.hpp"
#include <cmath>

// Initialize static member
std::array<int, 40> Entity::entityCounts = {};

Entity::Entity(int entityType, Simulation *sim, float xcoord, float ycoord)
    : entityType(entityType), sim(sim), xpos(xcoord), ypos(ycoord), xposOld(xcoord), yposOld(ycoord)
{
  // Initialize cell position
  cell = getCell();

  // Increment entity count for this type
  if (entityType >= 0 && entityType < 40)
  {
    entityCounts[entityType]++;
  }
}

std::vector<float> Entity::getState(bool minimalState) const
{
  std::vector<float> state;
  state.reserve(4);
  state.push_back(xpos);
  state.push_back(ypos);
  state.push_back(xspeed);
  state.push_back(yspeed);
  return state;
}

std::pair<int, int> Entity::getCell() const
{
  return {static_cast<int>(std::floor(xpos / 24)),
          static_cast<int>(std::floor(ypos / 24))};
}

void Entity::gridMove()
{
  auto newCell = getCell();
  if (newCell != cell)
  {
    // Remove from old cell
    auto &oldCellEntities = sim->getEntitiesAt(cell);
    auto it = std::find_if(oldCellEntities.begin(), oldCellEntities.end(),
                           [this](const auto &e)
                           { return e.get() == this; });
    if (it != oldCellEntities.end())
    {
      oldCellEntities.erase(it);
    }

    // Add to new cell
    sim->getEntitiesAt(newCell).push_back(std::shared_ptr<Entity>(this, [](Entity *) {})); // Non-owning shared_ptr
    cell = newCell;
  }
}

void Entity::logCollision(int state)
{
  collisionLog.push_back(state);
}

void Entity::logPosition()
{
  if (sim->getConfig().logData)
  {
    posLog.emplace_back(sim->getFrame(), xpos, ypos);
    speedLog.emplace_back(sim->getFrame(), xspeed, yspeed);
  }
}