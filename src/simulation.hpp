#pragma once

#include "sim_config.hpp"
#include "physics/grid_segment_linear.hpp"
#include "physics/grid_segment_circular.hpp"
#include "utils.hpp"

#include <unordered_map>
#include <vector>
#include <memory>
#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// Forward declarations
class Entity;
class Ninja;

using CellCoord = std::pair<int, int>;

class Simulation
{
public:
  // Grid data structures
  using SegmentList = std::vector<std::shared_ptr<Segment>>;
  using EntityList = std::vector<std::shared_ptr<Entity>>;

  // Tile map constants
  static const std::unordered_map<int, std::array<int, 12>> TILE_GRID_EDGE_MAP;
  static const std::unordered_map<int, std::array<int, 12>> TILE_SEGMENT_ORTHO_MAP;
  static const std::unordered_map<int, std::tuple<std::pair<int, int>, std::pair<int, int>>> TILE_SEGMENT_DIAG_MAP;
  static const std::unordered_map<int, std::tuple<std::pair<int, int>, std::pair<int, int>, bool>> TILE_SEGMENT_CIRCULAR_MAP;

  // Constructor
  explicit Simulation(const SimConfig &sc);

  // Map loading and reset methods
  void load(const std::vector<uint8_t> &mapData);
  void reset();

  // Main simulation update
  void tick(int horInput, int jumpInput);

  // Public accessors
  const Ninja *getNinja() const { return ninja.get(); }
  Ninja *getNinja() { return ninja.get(); }
  const SimConfig &getConfig() const { return simConfig; }
  int getFrame() const { return frame; }

  // Entity management
  std::shared_ptr<Entity> createEntity(int entityType, float xpos, float ypos, int orientation, int mode, float switchX = -1, float switchY = -1);
  void addEntity(std::shared_ptr<Entity> entity);
  void removeEntity(std::shared_ptr<Entity> entity);

  // Mutable accessors for entity management
  SegmentList &getSegmentsAt(const CellCoord &cell) { return segmentDic[cell]; }
  EntityList &getEntitiesAt(const CellCoord &cell) { return gridEntity[cell]; }
  EntityList &getEntitiesByType(int type) { return entityDic[type]; }
  const EntityList &getEntitiesOfType(int type) const { return entityDic.at(type); }

  // Const accessors for entity management
  const SegmentList &getSegmentsAt(const CellCoord &cell) const { return segmentDic.at(cell); }
  const EntityList &getEntitiesAt(const CellCoord &cell) const { return gridEntity.at(cell); }
  const EntityList &getEntitiesByType(int type) const { return entityDic.at(type); }

  // Grid edge accessors
  bool hasHorizontalEdge(const CellCoord &cell) const { return horGridEdgeDic.at(cell) != 0; }
  bool hasVerticalEdge(const CellCoord &cell) const { return verGridEdgeDic.at(cell) != 0; }

  // Map data accessors
  uint8_t getTileAt(int x, int y) const { return tileDic.at({x, y}); }

  // Add these accessor methods
  void incrementVerGridEdge(const std::pair<int, int> &edge, int amount) { verGridEdgeDic[edge] += amount; }
  void incrementHorGridEdge(const std::pair<int, int> &edge, int amount) { horGridEdgeDic[edge] += amount; }

  // Entity and segment gathering methods
  EntityList getEntitiesInRadius(float x, float y, float radius) const;
  SegmentList getSegmentsInRegion(float x1, float y1, float x2, float y2) const;

  // Add tile dictionary accessor
  const TileDictionary &getTileDic() const { return tileDic; }

private:
  // Internal map loading methods
  void resetMapEntityData();
  void resetMapTileData();
  void loadMapTiles();
  void loadMapEntities();

  // State variables
  int frame;
  std::vector<std::tuple<int, float, float>> collisionLog;
  SimConfig const &simConfig;
  std::unique_ptr<Ninja> ninja;

  // Map data structures
  std::unordered_map<CellCoord, int, CellCoordHash> tileDic;
  std::unordered_map<CellCoord, SegmentList, CellCoordHash> segmentDic;
  std::unordered_map<CellCoord, EntityList, CellCoordHash> gridEntity;
  std::unordered_map<int, EntityList> entityDic;
  std::unordered_map<CellCoord, int, CellCoordHash> horGridEdgeDic;
  std::unordered_map<CellCoord, int, CellCoordHash> verGridEdgeDic;
  std::unordered_map<CellCoord, int, CellCoordHash> horSegmentDic;
  std::unordered_map<CellCoord, int, CellCoordHash> verSegmentDic;
  std::vector<uint8_t> mapData;
};