#include "physics.hpp"
#include "segment.hpp"
#include "../simulation.hpp"
#include "../ninja.hpp"
#include "../entity.hpp"
#include "../entities/grid_segment_linear.hpp"
#include "../entities/grid_segment_circular.hpp"
#include <cmath>
#include <limits>
#include <sstream>
#include <optional>
#include <array>
#include <tuple>
#define M_PI 3.14159265358979323846

// Static member initialization
std::unordered_map<std::string, std::vector<std::pair<int, int>>> Physics::cellCache;
std::unordered_map<float, float> Physics::sqrtCache;

float Physics::clamp(float n, float a, float b)
{
  return n < a ? a : (n > b ? b : n);
}

std::pair<int, int> Physics::clampCell(int xcell, int ycell)
{
  return {clamp(xcell, 0, 43), clamp(ycell, 0, 24)};
}

std::pair<int, int> Physics::clampHalfCell(int xcell, int ycell)
{
  return {clamp(xcell, 0, 87), clamp(ycell, 0, 49)};
}

int16_t Physics::packCoord(float coord)
{
  constexpr int16_t lim = (1 << 15) - 1;
  return static_cast<int16_t>(clamp(std::round(10 * coord), -lim, lim));
}

float Physics::getCachedSqrt(float n)
{
  auto it = sqrtCache.find(n);
  if (it == sqrtCache.end())
  {
    it = sqrtCache.emplace(n, std::sqrt(n)).first;
  }
  return it->second;
}

void Physics::clearCaches()
{
  cellCache.clear();

  // Keep frequently used sqrt values
  sqrtCache.clear();
  const std::array<float, 12> commonValues = {0, 1, 2, 4, 9, 16, 25, 36, 49, 64, 81, 100};
  for (float v : commonValues)
  {
    sqrtCache[v] = std::sqrt(v);
  }
}

std::vector<std::tuple<float, float>> Physics::gatherSegmentsFromRegion(
    const Simulation &sim, float x1, float y1, float x2, float y2)
{
  // Create cache key
  std::stringstream key;
  key << std::floor(x1 / 24) << "," << std::floor(y1 / 24) << ","
      << std::floor(x2 / 24) << "," << std::floor(y2 / 24);

  std::vector<std::pair<int, int>> cells;
  auto cacheIt = cellCache.find(key.str());

  if (cacheIt != cellCache.end())
  {
    cells = cacheIt->second;
  }
  else
  {
    auto [cx1, cy1] = clampCell(std::floor(x1 / 24), std::floor(y1 / 24));
    auto [cx2, cy2] = clampCell(std::floor(x2 / 24), std::floor(y2 / 24));

    cells.reserve((cx2 - cx1 + 1) * (cy2 - cy1 + 1));
    for (int x = cx1; x <= cx2; ++x)
    {
      for (int y = cy1; y <= cy2; ++y)
      {
        cells.emplace_back(x, y);
      }
    }
    cellCache[key.str()] = cells;
  }

  std::vector<std::tuple<float, float>> segmentList;
  // TODO: Implement segment gathering from simulation
  return segmentList;
}

bool Physics::overlapCircleVsCircle(
    float xpos1, float ypos1, float radius1,
    float xpos2, float ypos2, float radius2)
{
  float dx = xpos1 - xpos2;
  float dy = ypos1 - ypos2;
  float distSq = dx * dx + dy * dy;
  float radiusSum = radius1 + radius2;
  return distSq < radiusSum * radiusSum;
}

float Physics::getTimeOfIntersectionCircleVsCircle(
    float xpos, float ypos, float vx, float vy,
    float a, float b, float radius)
{
  float dx = xpos - a;
  float dy = ypos - b;
  float distSq = dx * dx + dy * dy;
  float velSq = vx * vx + vy * vy;
  float dotProd = dx * vx + dy * vy;

  if (distSq - radius * radius > 0)
  {
    float radicand = dotProd * dotProd - velSq * (distSq - radius * radius);
    if (velSq > 0.0001f && dotProd < 0 && radicand >= 0)
    {
      return (-dotProd - std::sqrt(radicand)) / velSq;
    }
    return 1.0f;
  }
  return 0.0f;
}

float Physics::getTimeOfIntersectionCircleVsLineseg(
    float xpos, float ypos, float dx, float dy,
    float a1, float b1, float a2, float b2, float radius)
{
  float wx = a2 - a1;
  float wy = b2 - b1;
  float segLen = getCachedSqrt(wx * wx + wy * wy);

  if (segLen == 0)
    return 1.0f;

  float nx = wx / segLen;
  float ny = wy / segLen;
  float normalProj = (xpos - a1) * ny - (ypos - b1) * nx;
  float horProj = (xpos - a1) * nx + (ypos - b1) * ny;

  if (std::abs(normalProj) >= radius)
  {
    float dir = dx * ny - dy * nx;
    if (dir * normalProj < 0)
    {
      float t = std::min((std::abs(normalProj) - radius) / std::abs(dir), 1.0f);
      float horProj2 = horProj + t * (dx * nx + dy * ny);
      if (0 <= horProj2 && horProj2 <= segLen)
      {
        return t;
      }
    }
  }
  else if (0 <= horProj && horProj <= segLen)
  {
    return 0.0f;
  }
  return 1.0f;
}

float Physics::getTimeOfIntersectionCircleVsArc(
    float xpos, float ypos, float vx, float vy,
    float a, float b, float hor, float ver,
    float radiusArc, float radiusCircle)
{
  float dx = xpos - a;
  float dy = ypos - b;
  float distSq = dx * dx + dy * dy;
  float velSq = vx * vx + vy * vy;
  float dotProd = dx * vx + dy * vy;
  float radius1 = radiusArc + radiusCircle;
  float radius2 = radiusArc - radiusCircle;
  float t = 1.0f;

  if (distSq > radius1 * radius1)
  {
    float radicand = dotProd * dotProd - velSq * (distSq - radius1 * radius1);
    if (velSq > 0.0001f && dotProd < 0 && radicand >= 0)
    {
      t = (-dotProd - std::sqrt(radicand)) / velSq;
    }
  }
  else if (distSq < radius2 * radius2)
  {
    float radicand = dotProd * dotProd - velSq * (distSq - radius2 * radius2);
    if (velSq > 0.0001f)
    {
      t = std::min((-dotProd + std::sqrt(radicand)) / velSq, 1.0f);
    }
  }
  else
  {
    t = 0.0f;
  }

  if ((dx + t * vx) * hor > 0 && (dy + t * vy) * ver > 0)
  {
    return t;
  }
  return 1.0f;
}

std::pair<float, float> Physics::mapOrientationToVector(int orientation)
{
  static const std::array<std::pair<float, float>, 8> orientationMap = {{{1.0f, 0.0f},
                                                                         {0.70710678118f, 0.70710678118f},
                                                                         {0.0f, 1.0f},
                                                                         {-0.70710678118f, 0.70710678118f},
                                                                         {-1.0f, 0.0f},
                                                                         {-0.70710678118f, -0.70710678118f},
                                                                         {0.0f, -1.0f},
                                                                         {0.70710678118f, -0.70710678118f}}};

  return orientationMap[orientation % 8];
}

int Physics::mapVectorToOrientation(float xdir, float ydir)
{
  float angle = std::atan2(ydir, xdir);
  if (angle < 0)
  {
    angle += 2 * M_PI;
  }
  return static_cast<int>(std::round(8 * angle / (2 * M_PI))) % 8;
}

std::optional<float> Physics::getRaycastDistance(
    const Simulation &sim, float xpos, float ypos, float dx, float dy)
{
  int xcell = std::floor(xpos / 24);
  int ycell = std::floor(ypos / 24);

  float tmaxX, tmaxY, deltaX, deltaY;
  int stepX, stepY;

  if (dx > 0)
  {
    stepX = 1;
    deltaX = 24 / dx;
    tmaxX = ((xcell + 1) * 24 - xpos) / dx;
  }
  else if (dx < 0)
  {
    stepX = -1;
    deltaX = -24 / dx;
    tmaxX = (xcell * 24 - xpos) / dx;
  }
  else
  {
    stepX = 0;
    deltaX = 0;
    tmaxX = std::numeric_limits<float>::max();
  }

  if (dy > 0)
  {
    stepY = 1;
    deltaY = 24 / dy;
    tmaxY = ((ycell + 1) * 24 - ypos) / dy;
  }
  else if (dy < 0)
  {
    stepY = -1;
    deltaY = -24 / dy;
    tmaxY = (ycell * 24 - ypos) / dy;
  }
  else
  {
    stepY = 0;
    deltaY = 0;
    tmaxY = std::numeric_limits<float>::max();
  }

  float dist = 0;
  while (dist < 1)
  {
    if (tmaxX < tmaxY)
    {
      xcell += stepX;
      if (xcell < 0 || xcell >= 44)
        break;
      dist = tmaxX;
      tmaxX += deltaX;
    }
    else
    {
      ycell += stepY;
      if (ycell < 0 || ycell >= 25)
        break;
      dist = tmaxY;
      tmaxY += deltaY;
    }

    float cellDist = intersectRayVsCellContents(sim, xcell, ycell, xpos, ypos, dx, dy);
    if (cellDist < 1)
    {
      return cellDist;
    }
  }

  return std::nullopt;
}

bool Physics::raycastVsPlayer(
    const Simulation &sim, float xstart, float ystart,
    float ninjaXpos, float ninjaYpos, float ninjaRadius)
{
  float dx = ninjaXpos - xstart;
  float dy = ninjaYpos - ystart;
  float dist = getCachedSqrt(dx * dx + dy * dy);

  if (ninjaRadius <= dist && dist > 0)
  {
    dx /= dist;
    dy /= dist;
    auto length = getRaycastDistance(sim, xstart, ystart, dx, dy);
    return length && *length > dist - ninjaRadius;
  }
  return false;
}

bool Physics::overlapCircleVsSegment(
    float xpos, float ypos, float radius,
    float px1, float py1, float px2, float py2)
{
  float wx = px2 - px1;
  float wy = py2 - py1;
  float segLen = getCachedSqrt(wx * wx + wy * wy);

  if (segLen == 0)
    return false;

  float nx = wx / segLen;
  float ny = wy / segLen;
  float normalProj = (xpos - px1) * ny - (ypos - py1) * nx;
  float horProj = (xpos - px1) * nx + (ypos - py1) * ny;

  return std::abs(normalProj) < radius && 0 <= horProj && horProj <= segLen;
}

std::optional<std::tuple<std::pair<float, float>, std::pair<float, float>>>
Physics::penetrationSquareVsPoint(float sXpos, float sYpos, float pXpos, float pYpos, float semiSide)
{
  float dx = pXpos - sXpos;
  float dy = pYpos - sYpos;
  float penx = semiSide - std::abs(dx);
  float peny = semiSide - std::abs(dy);

  if (penx <= 0 || peny <= 0)
    return std::nullopt;

  std::pair<float, float> normal;
  if (penx < peny)
  {
    normal = {dx > 0 ? 1.0f : -1.0f, 0.0f};
    return std::make_tuple(normal, std::make_pair(peny, penx));
  }
  else
  {
    normal = {0.0f, dy > 0 ? 1.0f : -1.0f};
    return std::make_tuple(normal, std::make_pair(penx, peny));
  }
}

std::vector<std::tuple<float, float>> Physics::gatherEntitiesFromNeighbourhood(
    const Simulation &sim, float xpos, float ypos)
{
  auto [cx, cy] = clampCell(std::floor(xpos / 24), std::floor(ypos / 24));

  std::stringstream key;
  key << cx << "," << cy;

  std::vector<std::pair<int, int>> cells;
  auto cacheIt = cellCache.find(key.str());

  if (cacheIt != cellCache.end())
  {
    cells = cacheIt->second;
  }
  else
  {
    int minX = std::max(cx - 1, 0);
    int maxX = std::min(cx + 1, 43);
    int minY = std::max(cy - 1, 0);
    int maxY = std::min(cy + 1, 24);

    cells.reserve((maxX - minX + 1) * (maxY - minY + 1));
    for (int x = minX; x <= maxX; ++x)
    {
      for (int y = minY; y <= maxY; ++y)
      {
        cells.emplace_back(x, y);
      }
    }
    cellCache[key.str()] = cells;
  }

  std::vector<std::tuple<float, float>> entityList;

  for (const auto &cell : cells)
  {
    const auto &entities = sim.getEntitiesAt(cell);
    for (const auto &entity : entities)
    {
      if (entity->active)
      {
        entityList.emplace_back(entity->xpos, entity->ypos);
      }
    }
  }

  return entityList;
}

float Physics::sweepCircleVsTiles(
    const Simulation &sim, float xposOld, float yposOld,
    float dx, float dy, float radius)
{
  float t = 1.0f;

  // TODO: Implement tile collision detection
  return t;
}

std::tuple<int, std::optional<std::pair<float, float>>> Physics::getSingleClosestPoint(
    const Simulation &sim, float xpos, float ypos, float radius)
{
  // TODO: Implement closest point detection
  return {0, std::nullopt};
}

bool Physics::isEmptyRow(
    const Simulation &sim, int xcoord1, int xcoord2, int ycoord, int dir)
{
  // TODO: Implement row emptiness check
  return true;
}

bool Physics::isEmptyColumn(
    const Simulation &sim, int xcoord, int ycoord1, int ycoord2, int dir)
{
  // TODO: Implement column emptiness check
  return true;
}

bool Physics::checkLinesegVsNinja(
    float x1, float y1, float x2, float y2, const Ninja &ninja)
{
  // TODO: Implement line segment vs ninja collision check
  return false;
}

float Physics::intersectRayVsCellContents(
    const Simulation &sim, int xcell, int ycell,
    float xpos, float ypos, float dx, float dy)
{
  // TODO: Implement ray vs cell contents intersection
  return 1.0f;
}