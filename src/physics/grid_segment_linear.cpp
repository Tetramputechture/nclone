#include "grid_segment_linear.hpp"
#include "../physics/physics.hpp"
#include <cmath>

GridSegmentLinear::GridSegmentLinear(std::pair<float, float> p1, std::pair<float, float> p2, bool oriented)
    : Segment(p1.first, p1.second, p2.first, p2.second),
      x1(p1.first), y1(p1.second), x2(p2.first), y2(p2.second), oriented(oriented), active(true)
{
}

std::tuple<bool, float, float> GridSegmentLinear::getClosestPoint(float xpos, float ypos) const
{
  float px = x2 - x1;
  float py = y2 - y1;
  float dx = xpos - x1;
  float dy = ypos - y1;
  float segLenSq = px * px + py * py;
  float u = (dx * px + dy * py) / segLenSq;
  u = std::max(0.0f, std::min(1.0f, u));

  // If u is between 0 and 1, position is closest to the line segment.
  // If u is exactly 0 or 1, position is closest to one of the two edges.
  float a = x1 + u * px;
  float b = y1 + u * py;

  // Note: can't be backfacing if segment belongs to a door.
  bool isBackFacing = (dy * px - dx * py < 0) && oriented;

  return std::make_tuple(isBackFacing, a, b);
}

float GridSegmentLinear::intersectWithRay(float xpos, float ypos, float dx, float dy, float radius) const
{
  float time1 = Physics::getTimeOfIntersectionCircleVsCircle(xpos, ypos, dx, dy, x1, y1, radius);
  float time2 = Physics::getTimeOfIntersectionCircleVsCircle(xpos, ypos, dx, dy, x2, y2, radius);
  float time3 = Physics::getTimeOfIntersectionCircleVsLineseg(xpos, ypos, dx, dy, x1, y1, x2, y2, radius);

  return std::min({time1, time2, time3});
}