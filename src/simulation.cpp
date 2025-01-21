#include "simulation.hpp"
#include "ninja.hpp"
#include "physics/physics.hpp"
#include "entities/entity.hpp"
#include "physics/grid_segment_linear.hpp"
#include "physics/grid_segment_circular.hpp"
#include "entities/toggle_mine.hpp"
#include "entities/gold.hpp"
#include "entities/exit_door.hpp"
#include "entities/exit_switch.hpp"
#include "entities/door_regular.hpp"
#include "entities/door_locked.hpp"
#include "entities/door_trap.hpp"
#include "entities/launch_pad.hpp"
#include "entities/one_way_platform.hpp"
#include "entities/drone_zap.hpp"
#include "entities/drone_chaser.hpp"
#include "entities/bounce_block.hpp"
#include "entities/thwump.hpp"
#include "entities/laser.hpp"
#include "entities/boost_pad.hpp"
#include "entities/death_ball.hpp"
#include "entities/mini_drone.hpp"
#include "entities/shove_thwump.hpp"
#include <cmath>
#include <algorithm>

// Initialize static tile map constants
const std::unordered_map<int, std::array<int, 12>> Simulation::TILE_GRID_EDGE_MAP = {
    {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, // Empty tile
    {1, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}}, // Full tile
    {2, {1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0}}, // Half tiles
    {3, {0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1}},
    {4, {0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1}},
    {5, {1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0}},
    {6, {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}}, // 45 degree slopes
    {7, {1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1}},
    {8, {0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1}},
    {9, {1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1}},
    {10, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}}, // Quarter moons
    {11, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {12, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {13, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {14, {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}}, // Quarter pipes
    {15, {1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1}},
    {16, {0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1}},
    {17, {1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1}},
    {18, {1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0}}, // Short mild slopes
    {19, {1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0}},
    {20, {0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1}},
    {21, {0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1}},
    {22, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}}, // Raised mild slopes
    {23, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {24, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {25, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {26, {1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0}}, // Short steep slopes
    {27, {0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1}},
    {28, {0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1}},
    {29, {1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0}},
    {30, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}}, // Raised steep slopes
    {31, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {32, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {33, {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1}},
    {34, {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, // Glitched tiles
    {35, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
    {36, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0}},
    {37, {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}}};

const std::unordered_map<int, std::array<int, 12>> Simulation::TILE_SEGMENT_ORTHO_MAP = {
    {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},     // Empty tile
    {1, {-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1}}, // Full tile
    {2, {-1, -1, 1, 1, 0, 0, -1, 0, 0, 0, 1, 0}},  // Half tiles
    {3, {0, -1, 0, 0, 0, 1, 0, 0, -1, -1, 1, 1}},
    {4, {0, 0, -1, -1, 1, 1, 0, -1, 0, 0, 0, 1}},
    {5, {-1, 0, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0}},
    {6, {-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0}}, // 45 degree slopes
    {7, {-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
    {8, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}},
    {9, {0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0}},
    {10, {-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0}}, // Quarter moons
    {11, {-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
    {12, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}},
    {13, {0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0}},
    {14, {-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0}}, // Quarter pipes
    {15, {-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
    {16, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}},
    {17, {0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0}},
    {18, {-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0}}, // Short mild slopes
    {19, {-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}},
    {20, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1}},
    {21, {0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0}},
    {22, {-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0}}, // Raised mild slopes
    {23, {-1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1}},
    {24, {0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 1}},
    {25, {0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 1}},
    {26, {-1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0}}, // Short steep slopes
    {27, {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
    {28, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}},
    {29, {0, 0, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0}},
    {30, {-1, -1, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0}}, // Raised steep slopes
    {31, {-1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}},
    {32, {0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}},
    {33, {-1, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0}},
    {34, {-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, // Glitched tiles
    {35, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
    {36, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0}},
    {37, {0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0}}};

const std::unordered_map<int, std::tuple<std::pair<int, int>, std::pair<int, int>, bool>> Simulation::TILE_SEGMENT_CIRCULAR_MAP = {
    {10, {{0, 0}, {1, 1}, true}}, // Quarter moons
    {11, {{24, 0}, {-1, 1}, true}},
    {12, {{24, 24}, {-1, -1}, true}},
    {13, {{0, 24}, {1, -1}, true}},
    {14, {{24, 24}, {-1, -1}, false}}, // Quarter pipes
    {15, {{0, 24}, {1, -1}, false}},
    {16, {{0, 0}, {1, 1}, false}},
    {17, {{24, 0}, {-1, 1}, false}}};

const std::unordered_map<int, std::tuple<std::pair<int, int>, std::pair<int, int>>> Simulation::TILE_SEGMENT_DIAG_MAP = {
    {6, {{0, 24}, {24, 0}}}, // 45 degree slopes
    {7, {{0, 0}, {24, 24}}},
    {8, {{24, 0}, {0, 24}}},
    {9, {{24, 24}, {0, 0}}},
    {18, {{0, 12}, {24, 0}}}, // Short mild slopes
    {19, {{0, 0}, {24, 12}}},
    {20, {{24, 12}, {0, 24}}},
    {21, {{24, 24}, {0, 12}}},
    {22, {{0, 24}, {24, 12}}}, // Raised mild slopes
    {23, {{0, 12}, {24, 24}}},
    {24, {{24, 0}, {0, 12}}},
    {25, {{24, 12}, {0, 0}}},
    {26, {{0, 24}, {12, 0}}}, // Short steep slopes
    {27, {{12, 0}, {24, 24}}},
    {28, {{24, 0}, {12, 24}}},
    {29, {{12, 24}, {0, 0}}},
    {30, {{12, 24}, {24, 0}}}, // Raised steep slopes
    {31, {{0, 0}, {12, 24}}},
    {32, {{12, 0}, {0, 24}}},
    {33, {{24, 24}, {12, 0}}}};

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

  // Reset entity counts
  std::fill(Entity::entityCounts.begin(), Entity::entityCounts.end(), 0);

  // Process entity data
  size_t index = 1234;
  size_t exitDoorCount = mapData[1156];

  while (index < mapData.size())
  {
    int entityType = mapData[index];
    if (entityType == 0)
      break;

    float xpos = static_cast<float>(mapData[index + 1] + (mapData[index + 2] << 8)) / 10.0f;
    float ypos = static_cast<float>(mapData[index + 3] + (mapData[index + 4] << 8)) / 10.0f;
    int orientation = mapData[index + 3];
    int mode = mapData[index + 4];

    float switchX = -1;
    float switchY = -1;

    // Handle switch coordinates for doors
    if (entityType == 6 || entityType == 8)
    {
      switchX = static_cast<float>(mapData[index + 6]);
      switchY = static_cast<float>(mapData[index + 7]);
    }
    // Handle exit door switch coordinates
    else if (entityType == 3)
    {
      switchX = static_cast<float>(mapData[index + 5 * exitDoorCount + 1]);
      switchY = static_cast<float>(mapData[index + 5 * exitDoorCount + 2]);
    }

    auto entity = createEntity(entityType, xpos, ypos, orientation, mode, switchX, switchY);
    if (entity)
    {
      addEntity(entity);
      entity->logPosition();
    }

    index += 5;
  }
}

std::shared_ptr<Entity> Simulation::createEntity(int entityType, float xpos, float ypos, int orientation, int mode, float switchX, float switchY)
{
  std::shared_ptr<Entity> entity;

  switch (entityType)
  {
  case 1: // Toggle Mine
    entity = std::static_pointer_cast<Entity>(std::make_shared<ToggleMine>(this, xpos, ypos, 0));
    break;
  case 2: // Gold
    entity = std::static_pointer_cast<Entity>(std::make_shared<Gold>(this, xpos, ypos));
    break;
  case 3:
  { // Exit Door
    auto exitDoor = std::make_shared<ExitDoor>(this, xpos, ypos);
    auto &typeList = entityDic[entityType];
    typeList.push_back(std::static_pointer_cast<Entity>(exitDoor));

    // Create and return the exit switch using provided coordinates
    entity = std::static_pointer_cast<Entity>(std::make_shared<ExitSwitch>(this, switchX, switchY, exitDoor.get()));
    break;
  }
  case 5: // Regular Door
    entity = std::static_pointer_cast<Entity>(std::make_shared<DoorRegular>(this, xpos, ypos, orientation, xpos, ypos));
    break;
  case 6: // Locked Door
    entity = std::static_pointer_cast<Entity>(std::make_shared<DoorLocked>(this, xpos, ypos, orientation, switchX, switchY));
    break;
  case 8: // Trap Door
    entity = std::static_pointer_cast<Entity>(std::make_shared<DoorTrap>(this, xpos, ypos, orientation, switchX, switchY));
    break;
  case 10: // Launch Pad
    entity = std::static_pointer_cast<Entity>(std::make_shared<LaunchPad>(this, xpos, ypos, orientation));
    break;
  case 11: // One Way Platform
    entity = std::static_pointer_cast<Entity>(std::make_shared<OneWayPlatform>(this, xpos, ypos, orientation));
    break;
  case 14: // Drone Zap
    if (!simConfig.basicSim)
    {
      entity = std::static_pointer_cast<Entity>(std::make_shared<DroneZap>(this, xpos, ypos, orientation, mode));
    }
    break;
  case 15: // Drone Chaser
    if (!simConfig.basicSim)
    {
      entity = std::static_pointer_cast<Entity>(std::make_shared<DroneChaser>(this, xpos, ypos, orientation, mode));
    }
    break;
  case 17: // Bounce Block
    entity = std::static_pointer_cast<Entity>(std::make_shared<BounceBlock>(this, xpos, ypos));
    break;
  case 20: // Thwump
    entity = std::static_pointer_cast<Entity>(std::make_shared<Thwump>(this, xpos, ypos, orientation));
    break;
  case 21: // Toggle Mine (toggled state)
    entity = std::static_pointer_cast<Entity>(std::make_shared<ToggleMine>(this, xpos, ypos, 1));
    break;
  case 23: // Laser
    if (!simConfig.basicSim)
    {
      entity = std::static_pointer_cast<Entity>(std::make_shared<Laser>(this, xpos, ypos, orientation, mode));
    }
    break;
  case 24: // Boost Pad
    entity = std::static_pointer_cast<Entity>(std::make_shared<BoostPad>(this, xpos, ypos));
    break;
  case 25: // Death Ball
    if (!simConfig.basicSim)
    {
      entity = std::static_pointer_cast<Entity>(std::make_shared<DeathBall>(this, xpos, ypos));
    }
    break;
  case 26: // Mini Drone
    if (!simConfig.basicSim)
    {
      entity = std::static_pointer_cast<Entity>(std::make_shared<MiniDrone>(this, xpos, ypos, orientation, mode));
    }
    break;
  case 28: // Shove Thwump
    entity = std::static_pointer_cast<Entity>(std::make_shared<ShoveThwump>(this, xpos, ypos));
    break;
  }

  return entity;
}

void Simulation::addEntity(std::shared_ptr<Entity> entity)
{
  if (!entity)
    return;

  int type = entity->getType();
  auto cell = entity->getCell();

  entityDic[type].push_back(entity);
  gridEntity[cell].push_back(entity);
}

void Simulation::removeEntity(std::shared_ptr<Entity> entity)
{
  if (!entity)
    return;

  int type = entity->getType();
  auto cell = entity->getCell();

  // Remove from type dictionary
  auto &typeList = entityDic[type];
  typeList.erase(std::remove(typeList.begin(), typeList.end(), entity), typeList.end());

  // Remove from grid
  auto &cellList = gridEntity[cell];
  cellList.erase(std::remove(cellList.begin(), cellList.end(), entity), cellList.end());
}

void Simulation::tick(float horInput, int jumpInput)
{
  // Increment the current frame
  frame++;

  // Store inputs as ninja variables
  ninja->setHorInput(horInput);
  ninja->setJumpInput(jumpInput);

  // Cache active entities to avoid repeated filtering
  std::vector<std::shared_ptr<Entity>> activeMovableEntities;
  std::vector<std::shared_ptr<Entity>> activeThinkableEntities;

  // Single pass to categorize entities
  for (const auto &[type, entityList] : entityDic)
  {
    for (const auto &entity : entityList)
    {
      if (!entity->isActive())
        continue;

      if (entity->isMovable())
      {
        activeMovableEntities.push_back(entity);
      }
      if (entity->isThinkable())
      {
        activeThinkableEntities.push_back(entity);
      }
    }
  }

  // Move all movable entities
  for (const auto &entity : activeMovableEntities)
  {
    entity->move();
  }

  // Make all thinkable entities think
  for (const auto &entity : activeThinkableEntities)
  {
    entity->think();
  }

  if (ninja->getState() != 9)
  {
    ninja->integrate();    // Do preliminary speed and position updates
    ninja->preCollision(); // Do pre collision calculations

    // Cache collision results
    for (int i = 0; i < 4; i++)
    {
      // Handle PHYSICAL collisions with entities
      ninja->collideVsObjects(*this);
      // Handle physical collisions with tiles
      ninja->collideVsTiles(*this);
    }

    ninja->postCollision(*this); // Do post collision calculations
    ninja->think();              // Make ninja think

    if (simConfig.enableAnim)
    {
      ninja->updateGraphics(); // Update limbs of ninja
    }
  }

  if (ninja->getState() == 6 && simConfig.enableAnim)
  {
    // Placeholder because no ragdoll!
    ninja->setAnimFrame(105);
    ninja->setAnimState(7);
    ninja->calcNinjaPosition();
  }

  if (simConfig.logData)
  {
    // Update all the logs for debugging purposes and for tracing the route
    ninja->log(frame);

    // Batch entity position logging
    for (const auto &entity : activeMovableEntities)
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

Simulation::EntityList Simulation::getEntitiesInRadius(float x, float y, float radius) const
{
  EntityList result;

  // Calculate grid cell range to check
  int minCellX = static_cast<int>((x - radius) / 6.0f);
  int maxCellX = static_cast<int>((x + radius) / 6.0f);
  int minCellY = static_cast<int>((y - radius) / 6.0f);
  int maxCellY = static_cast<int>((y + radius) / 6.0f);

  // Gather entities from each cell in range
  for (int cellX = minCellX; cellX <= maxCellX; ++cellX)
  {
    for (int cellY = minCellY; cellY <= maxCellY; ++cellY)
    {
      CellCoord cell{cellX, cellY};
      auto it = gridEntity.find(cell);
      if (it != gridEntity.end())
      {
        for (const auto &entity : it->second)
        {
          float dx = entity->getXPos() - x;
          float dy = entity->getYPos() - y;
          if (dx * dx + dy * dy <= radius * radius)
          {
            result.push_back(entity);
          }
        }
      }
    }
  }

  return result;
}

Simulation::SegmentList Simulation::getSegmentsInRegion(float x1, float y1, float x2, float y2) const
{
  SegmentList result;

  // Calculate grid cell range to check
  int minCellX = static_cast<int>(std::min(x1, x2) / 6.0f);
  int maxCellX = static_cast<int>(std::max(x1, x2) / 6.0f);
  int minCellY = static_cast<int>(std::min(y1, y2) / 6.0f);
  int maxCellY = static_cast<int>(std::max(y1, y2) / 6.0f);

  // Gather segments from each cell in range
  for (int cellX = minCellX; cellX <= maxCellX; ++cellX)
  {
    for (int cellY = minCellY; cellY <= maxCellY; ++cellY)
    {
      CellCoord cell{cellX, cellY};
      auto it = segmentDic.find(cell);
      if (it != segmentDic.end())
      {
        result.insert(result.end(), it->second.begin(), it->second.end());
      }
    }
  }

  return result;
}