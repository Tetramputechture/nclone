#include "physics.hpp"
#include "../simulation.hpp"
#include "../entities/entity.hpp"
#include "segment.hpp"
#include "grid_segment_linear.hpp"
#include "grid_segment_circular.hpp"
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

std::vector<const Segment *> Physics::gatherSegmentsFromRegion(
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

  std::vector<const Segment *> segmentList;
  for (const auto &cell : cells)
  {
    const auto &segments = sim.getSegmentsAt(cell);
    for (const auto &segment : segments)
    {
      if (segment->isActive())
      {
        segmentList.push_back(segment.get());
      }
    }
  }
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

  if (penx > 0 && peny > 0)
  {
    if (peny <= penx)
    {
      std::pair<float, float> depenNormal{0.0f, dy < 0 ? -1.0f : 1.0f};
      std::pair<float, float> depenValues{peny, penx};
      return std::make_tuple(depenNormal, depenValues);
    }
    else
    {
      std::pair<float, float> depenNormal{dx < 0 ? -1.0f : 1.0f, 0.0f};
      std::pair<float, float> depenValues{penx, peny};
      return std::make_tuple(depenNormal, depenValues);
    }
  }
  return std::nullopt;
}

std::vector<Entity *> Physics::gatherEntitiesFromNeighbourhood(
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

  std::vector<Entity *> entityList;

  for (const auto &cell : cells)
  {
    const auto &entities = sim.getEntitiesAt(cell);
    for (const auto &entity : entities)
    {
      if (entity->isActive())
      {
        entityList.push_back(entity.get());
      }
    }
  }

  return entityList;
}

float Physics::sweepCircleVsTiles(
    const Simulation &sim, float xposOld, float yposOld,
    float dx, float dy, float radius)
{
  float xposNew = xposOld + dx;
  float yposNew = yposOld + dy;
  float width = radius + 1;
  float x1 = std::min(xposOld, xposNew) - width;
  float y1 = std::min(yposOld, yposNew) - width;
  float x2 = std::max(xposOld, xposNew) + width;
  float y2 = std::max(yposOld, yposNew) + width;

  auto segments = gatherSegmentsFromRegion(sim, x1, y1, x2, y2);
  float shortestTime = 1.0f;

  for (const auto &segment : segments)
  {
    float time = segment->intersectWithRay(xposOld, yposOld, dx, dy, radius);
    shortestTime = std::min(time, shortestTime);
  }

  return shortestTime;
}

std::optional<std::tuple<bool, std::pair<float, float>>>
Physics::getSingleClosestPoint(const Simulation &sim, float xpos, float ypos, float radius)
{
  auto segments = gatherSegmentsFromRegion(sim, xpos - radius, ypos - radius, xpos + radius, ypos + radius);
  float shortestDistance = std::numeric_limits<float>::infinity();
  bool result = false;
  std::pair<float, float> closestPoint;

  for (const auto &segment : segments)
  {
    auto [isBackFacing, a, b] = segment->getClosestPoint(xpos, ypos);
    float distanceSq = (xpos - a) * (xpos - a) + (ypos - b) * (ypos - b);

    // This is to prioritize correct side collisions when multiple close segments
    if (!isBackFacing)
    {
      distanceSq -= 0.1f;
    }

    if (distanceSq < shortestDistance)
    {
      shortestDistance = distanceSq;
      closestPoint = {a, b};
      result = isBackFacing;
    }
  }

  if (shortestDistance == std::numeric_limits<float>::infinity())
  {
    return std::nullopt;
  }

  return std::make_tuple(result, closestPoint);
}

bool Physics::isEmptyRow(
    const Simulation &sim, int xcoord1, int xcoord2, int ycoord, int dir)
{
  for (int xcoord = xcoord1; xcoord <= xcoord2; ++xcoord)
  {
    if (dir == 1)
    {
      auto [x, y] = clampHalfCell(xcoord, ycoord + 1);
      if (sim.hasHorizontalEdge({x, y}))
        return false;
    }
    else if (dir == -1)
    {
      auto [x, y] = clampHalfCell(xcoord, ycoord);
      if (sim.hasHorizontalEdge({x, y}))
        return false;
    }
  }
  return true;
}

bool Physics::isEmptyColumn(
    const Simulation &sim, int xcoord, int ycoord1, int ycoord2, int dir)
{
  for (int ycoord = ycoord1; ycoord <= ycoord2; ++ycoord)
  {
    if (dir == 1)
    {
      auto [x, y] = clampHalfCell(xcoord + 1, ycoord);
      if (sim.hasVerticalEdge({x, y}))
        return false;
    }
    else if (dir == -1)
    {
      auto [x, y] = clampHalfCell(xcoord, ycoord);
      if (sim.hasVerticalEdge({x, y}))
        return false;
    }
  }
  return true;
}

bool Physics::checkLinesegVsNinja(
    float x1, float y1, float x2, float y2, const Ninja &ninja)
{
  float dx = x2 - x1;
  float dy = y2 - y1;
  float len = getCachedSqrt(dx * dx + dy * dy);
  if (len == 0)
    return false;

  // This part returns false if the segment does not intersect the ninja's circular hitbox
  dx /= len;
  dy /= len;
  float proj = (ninja.xpos - x1) * dx + (ninja.ypos - y1) * dy;
  float x = x1;
  float y = y1;
  if (proj > 0)
  {
    x += dx * proj;
    y += dy * proj;
  }
  if (ninja.RADIUS * ninja.RADIUS <= (ninja.xpos - x) * (ninja.xpos - x) + (ninja.ypos - y) * (ninja.ypos - y))
    return false;

  // Now test the segment against each of ninja's 11 segments
  static const std::array<std::pair<int, int>, 11> NINJA_SEGS = {
      std::make_pair(0, 12),
      std::make_pair(1, 12),
      std::make_pair(2, 8),
      std::make_pair(3, 9),
      std::make_pair(4, 10),
      std::make_pair(5, 11),
      std::make_pair(6, 7),
      std::make_pair(8, 0),
      std::make_pair(9, 0),
      std::make_pair(10, 1),
      std::make_pair(11, 1)};

  for (const auto &seg : NINJA_SEGS)
  {
    const auto &bone1 = ninja.bones[seg.first];
    const auto &bone2 = ninja.bones[seg.second];
    float x3 = ninja.xpos + 24 * bone1.first;
    float y3 = ninja.ypos + 24 * bone1.second;
    float x4 = ninja.xpos + 24 * bone2.first;
    float y4 = ninja.ypos + 24 * bone2.second;

    float det1 = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3);
    float det2 = (x1 - x4) * (y2 - y4) - (y1 - y4) * (x2 - x4);
    float det3 = (x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1);
    float det4 = (x3 - x2) * (y4 - y2) - (y3 - y2) * (x4 - x2);

    if (det1 * det2 < 0 && det3 * det4 < 0)
      return true;
  }
  return false;
}

/**
 * Given a cell and a ray, return the shortest time of intersection between the ray and one of
 * the cell's tile segments. Return 1 if the ray hits nothing.
 */
float Physics::intersectRayVsCellContents(
    const Simulation &sim, int xcell, int ycell,
    float xpos, float ypos, float dx, float dy)
{
  auto segments = sim.getSegmentsAt(clampCell(xcell, ycell));
  float shortestTime = 1.0f;
  for (const auto &segment : segments)
  {
    float time = segment->intersectWithRay(xpos, ypos, dx, dy, 0);
    shortestTime = std::min(time, shortestTime);
  }
  return shortestTime;
}
