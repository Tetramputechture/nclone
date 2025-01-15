#include "simulation.hpp"
#include "ninja.hpp"
#include "sim_config.hpp"
#include "entities/grid_segment_linear.hpp"
#include "entities/grid_segment_circular.hpp"
#include <cmath>

// Initialize static tile map constants
const std::unordered_map<int, std::array<int, 12>> Simulation::TILE_GRID_EDGE_MAP = {
    {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {1, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {2, {1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0}},
    {3, {0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1}},
    {4, {0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1}},
    {5, {1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0}},
    {6, {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}},
    {7, {1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1}},
    {8, {0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1}},
    {9, {1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1}},
    // Add remaining tile mappings...
};

const std::unordered_map<int, std::array<int, 12>> Simulation::TILE_SEGMENT_ORTHO_MAP = {
    {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {1, {-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1}},
    {2, {-1, -1, 1, 1, 0, 0, -1, 0, 0, 0, 1, 0}},
    {3, {0, -1, 0, 0, 0, 1, 0, 0, -1, -1, 1, 1}},
    {4, {0, 0, -1, -1, 1, 1, 0, -1, 0, 0, 0, 1}},
    {5, {-1, 0, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0}},
    // Add remaining tile mappings...
};

const std::unordered_map<int, std::tuple<std::pair<int, int>, std::pair<int, int>>> Simulation::TILE_SEGMENT_DIAG_MAP = {
    {6, {{0, 24}, {24, 0}}},
    {7, {{0, 0}, {24, 24}}},
    {8, {{24, 0}, {0, 24}}},
    {9, {{24, 24}, {0, 0}}},
    // Add remaining diagonal mappings...
};

const std::unordered_map<int, std::tuple<std::pair<int, int>, std::pair<int, int>, bool>> Simulation::TILE_SEGMENT_CIRCULAR_MAP = {
    {10, {{0, 0}, {1, 1}, true}},
    {11, {{24, 0}, {-1, 1}, true}},
    {12, {{24, 24}, {-1, -1}, true}},
    {13, {{0, 24}, {1, -1}, true}},
    // Add remaining circular mappings...
};

Simulation::Simulation(const SimConfig &sc)
    : frame(0), simConfig(sc), ninja(nullptr)
{
}

void Simulation::resetMapEntityData()
{
  gridEntity.clear();
  entityDic.clear();

  // Initialize grid cells
  for (int x = 0; x < 44; ++x)
  {
    for (int y = 0; y < 25; ++y)
    {
      gridEntity[{x, y}] = EntityList();
    }
  }

  // Initialize entity type lists
  for (int i = 1; i < 29; ++i)
  {
    entityDic[i] = EntityList();
  }
}

void Simulation::resetMapTileData()
{
  segmentDic.clear();
  horGridEdgeDic.clear();
  verGridEdgeDic.clear();
  horSegmentDic.clear();
  verSegmentDic.clear();

  // Initialize segment dictionary
  for (int x = 0; x < 45; ++x)
  {
    for (int y = 0; y < 26; ++y)
    {
      segmentDic[{x, y}] = SegmentList();
    }
  }

  // Initialize grid edges
  for (int x = 0; x < 89; ++x)
  {
    for (int y = 0; y < 51; ++y)
    {
      horGridEdgeDic[{x, y}] = (y == 0 || y == 50) ? 1 : 0;
      verGridEdgeDic[{x, y}] = (x == 0 || x == 88) ? 1 : 0;
      horSegmentDic[{x, y}] = 0;
      verSegmentDic[{x, y}] = 0;
    }
  }
}

void Simulation::load(const std::vector<uint8_t> &mapData)
{
  this->mapData = mapData;
  resetMapTileData();
  loadMapTiles();
  reset();
}

void Simulation::reset()
{
  frame = 0;
  collisionLog.clear();
  ninja.reset();
  resetMapEntityData();
  loadMapEntities();
}

void Simulation::loadMapTiles()
{
  // Extract tile data from map data
  auto tileData = std::vector<uint8_t>(mapData.begin() + 184, mapData.begin() + 1150);

  // Map each tile to its cell
  for (int x = 0; x < 42; ++x)
  {
    for (int y = 0; y < 23; ++y)
    {
      tileDic[{x + 1, y + 1}] = tileData[x + y * 42];
    }
  }

  // Set outer edges to tile type 1 (full tile)
  for (int x = 0; x < 44; ++x)
  {
    tileDic[{x, 0}] = 1;
    tileDic[{x, 24}] = 1;
  }
  for (int y = 0; y < 25; ++y)
  {
    tileDic[{0, y}] = 1;
    tileDic[{43, y}] = 1;
  }

  // Process each tile
  for (const auto &[coord, tileId] : tileDic)
  {
    auto [xcoord, ycoord] = coord;
    int xtl = xcoord * 24;
    int ytl = ycoord * 24;

    auto gridEdgeIter = TILE_GRID_EDGE_MAP.find(tileId);
    auto segmentOrthoIter = TILE_SEGMENT_ORTHO_MAP.find(tileId);

    if (gridEdgeIter != TILE_GRID_EDGE_MAP.end() && segmentOrthoIter != TILE_SEGMENT_ORTHO_MAP.end())
    {
      const auto &gridEdgeList = gridEdgeIter->second;
      const auto &segmentOrthoList = segmentOrthoIter->second;

      // Process horizontal edges and segments
      for (int y = 0; y < 3; ++y)
      {
        for (int x = 0; x < 2; ++x)
        {
          CellCoord pos{2 * xcoord + x, 2 * ycoord + y};
          horGridEdgeDic[pos] = (horGridEdgeDic[pos] + gridEdgeList[2 * y + x]) % 2;
          horSegmentDic[pos] += segmentOrthoList[2 * y + x];
        }
      }

      // Process vertical edges and segments
      for (int x = 0; x < 3; ++x)
      {
        for (int y = 0; y < 2; ++y)
        {
          CellCoord pos{2 * xcoord + x, 2 * ycoord + y};
          verGridEdgeDic[pos] = (verGridEdgeDic[pos] + gridEdgeList[2 * x + y + 6]) % 2;
          verSegmentDic[pos] += segmentOrthoList[2 * x + y + 6];
        }
      }
    }

    // Process diagonal segments
    auto diagIter = TILE_SEGMENT_DIAG_MAP.find(tileId);
    if (diagIter != TILE_SEGMENT_DIAG_MAP.end())
    {
      const auto &[p1, p2] = diagIter->second;
      segmentDic[coord].push_back(std::make_shared<GridSegmentLinear>(
          std::make_pair(xtl + p1.first, ytl + p1.second),
          std::make_pair(xtl + p2.first, ytl + p2.second)));
    }

    // Process circular segments
    auto circIter = TILE_SEGMENT_CIRCULAR_MAP.find(tileId);
    if (circIter != TILE_SEGMENT_CIRCULAR_MAP.end())
    {
      const auto &[center, quadrant, convex] = circIter->second;
      segmentDic[coord].push_back(std::make_shared<GridSegmentCircular>(
          std::make_pair(xtl + center.first, ytl + center.second),
          quadrant, convex));
    }
  }

  // Create segments from horizontal segment dictionary
  for (const auto &[coord, state] : horSegmentDic)
  {
    if (state)
    {
      auto [xcoord, ycoord] = coord;
      CellCoord cell{static_cast<int>(std::floor(xcoord / 2)),
                     static_cast<int>(std::floor((ycoord - 0.1f * state) / 2))};

      std::pair<float, float> point1{12 * xcoord, 12 * ycoord};
      std::pair<float, float> point2{12 * xcoord + 12, 12 * ycoord};

      if (state == -1)
      {
        std::swap(point1, point2);
      }

      segmentDic[cell].push_back(std::make_shared<GridSegmentLinear>(point1, point2));
    }
  }

  // Create segments from vertical segment dictionary
  for (const auto &[coord, state] : verSegmentDic)
  {
    if (state)
    {
      auto [xcoord, ycoord] = coord;
      CellCoord cell{static_cast<int>(std::floor((xcoord - 0.1f * state) / 2)),
                     static_cast<int>(std::floor(ycoord / 2))};

      std::pair<float, float> point1{12 * xcoord, 12 * ycoord + 12};
      std::pair<float, float> point2{12 * xcoord, 12 * ycoord};

      if (state == -1)
      {
        std::swap(point1, point2);
      }

      segmentDic[cell].push_back(std::make_shared<GridSegmentLinear>(point1, point2));
    }
  }
}

void Simulation::loadMapEntities()
{
  // Create player ninja
  ninja = std::make_unique<Ninja>();

  // Force map data[1233] to be -1 if not valid
  if (mapData[1233] != -1 && mapData[1233] != 1)
  {
    mapData[1233] = -1;
  }

  // Process entity data
  size_t index = 1234;
  while (index < mapData.size())
  {
    int entityType = mapData[index];
    if (entityType == 0)
      break;

    float xpos = static_cast<float>(mapData[index + 1] + (mapData[index + 2] << 8)) / 10.0f;
    float ypos = static_cast<float>(mapData[index + 3] + (mapData[index + 4] << 8)) / 10.0f;

    auto entity = createEntity(entityType, xpos, ypos);
    if (entity)
    {
      entityDic[entityType].push_back(entity);
      gridEntity[entity->getCell()].push_back(entity);
      entity->logPosition();
    }

    index += 5;
  }
}

void Simulation::tick(float horInput, int jumpInput)
{
  frame++;

  // Update ninja inputs
  if (ninja)
  {
    ninja->horInput = horInput;
    ninja->jumpInput = jumpInput;
  }

  // Cache active entities
  std::vector<Entity *> activeMovableEntities;
  std::vector<Entity *> activeThinkableEntities;

  // Gather active entities
  for (const auto &[_, entityList] : entityDic)
  {
    for (const auto &entity : entityList)
    {
      if (!entity->active)
        continue;

      if (entity->isMovable())
      {
        activeMovableEntities.push_back(entity.get());
      }
      if (entity->isThinkable())
      {
        activeThinkableEntities.push_back(entity.get());
      }
    }
  }

  // Update movable entities
  for (auto entity : activeMovableEntities)
  {
    entity->move();
  }

  // Update thinkable entities
  for (auto entity : activeThinkableEntities)
  {
    entity->think();
  }

  // Update ninja physics
  if (ninja && ninja->state != 9)
  {
    auto physicsTarget = ninja.get();

    if (physicsTarget)
    {
      physicsTarget->integrate();
      physicsTarget->preCollision();

      for (int i = 0; i < 4; i++)
      {
        physicsTarget->collideVsObjects();
        physicsTarget->collideVsTiles();
      }

      physicsTarget->postCollision();
      ninja->think();

      if (simConfig.enableAnim)
      {
        ninja->updateGraphics();
      }
    }
  }

  // Handle dead ninja animation
  if (ninja && ninja->state == 6 && simConfig.enableAnim)
  {
    ninja->animFrame = 105;
    ninja->animState = 7;
    ninja->updateGraphics();
  }

  // Log data if enabled
  if (simConfig.logData)
  {
    if (ninja)
    {
      ninja->log(frame);
    }

    for (auto entity : activeMovableEntities)
    {
      entity->logPosition();
    }
  }

  // Clear physics caches periodically
  if (frame % 100 == 0)
  {
    Physics::clearCaches();
  }
}