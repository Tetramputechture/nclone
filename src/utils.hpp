#pragma once

#include <utility>
#include <cmath>
#include <unordered_map>

// Hash function for coordinate pairs
struct CellCoordHash
{
  std::size_t operator()(const std::pair<int, int> &coord) const
  {
    return std::hash<int>()(coord.first) ^ (std::hash<int>()(coord.second) << 1);
  }
};

// Type alias for tile dictionary
using TileDictionary = std::unordered_map<std::pair<int, int>, int, CellCoordHash>;

// Convert orientation (0-7) to a normalized vector
inline std::pair<float, float> mapOrientationToVector(int orientation)
{
  switch (orientation)
  {
  case 0:
  case 1:
    return {1.0f, 0.0f};
  case 2:
  case 3:
    return {0.0f, 1.0f};
  case 4:
  case 5:
    return {-1.0f, 0.0f};
  case 6:
  case 7:
  default:
    return {0.0f, -1.0f};
  }
}

// Clamp cell coordinates to valid range
inline std::pair<int, int> clampCell(int x, int y)
{
  // Assuming 44x25 grid size from Python code
  return {
      std::max(0, std::min(x, 43)),
      std::max(0, std::min(y, 24))};
}