#include "grid_segment_circular.hpp"
#include "../physics/physics.hpp"
#include <cmath>

GridSegmentCircular::GridSegmentCircular(std::pair<float, float> center, std::pair<int, int> quadrant, bool convex, float radius)
    : Segment(center.first, center.second, center.first + radius * quadrant.first, center.second + radius * quadrant.second),
      xpos(center.first), ypos(center.second),
      hor(quadrant.first), ver(quadrant.second),
      radius(radius), active(true), convex(convex)
{
  // Calculate the positions of the two extremities of arc
  pHor = std::make_pair(xpos + radius * hor, ypos);
  pVer = std::make_pair(xpos, ypos + radius * ver);
}

std::tuple<bool, float, float> GridSegmentCircular::getClosestPoint(float xposIn, float yposIn) const
{
  float dx = xposIn - xpos;
  float dy = yposIn - ypos;
  bool isBackFacing = false;

  // This is true if position is closer from arc than its edges
  if (dx * hor > 0 && dy * ver > 0)
  {
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > 0)
    {
      float a = xpos + radius * dx / dist;
      float b = ypos + radius * dy / dist;
      isBackFacing = dist < radius ? convex : !convex;
      return std::make_tuple(isBackFacing, a, b);
    }
  }

  // If closer to edges of arc, find position of closest point of the two
  if (dx * hor > dy * ver)
  {
    return std::make_tuple(isBackFacing, pHor.first, pHor.second);
  }
  else
  {
    return std::make_tuple(isBackFacing, pVer.first, pVer.second);
  }
}

float GridSegmentCircular::intersectWithRay(float xposIn, float yposIn, float dx, float dy, float radiusIn) const
{
  float time1 = Physics::getTimeOfIntersectionCircleVsCircle(xposIn, yposIn, dx, dy, pHor.first, pHor.second, radiusIn);
  float time2 = Physics::getTimeOfIntersectionCircleVsCircle(xposIn, yposIn, dx, dy, pVer.first, pVer.second, radiusIn);
  float time3 = Physics::getTimeOfIntersectionCircleVsArc(xposIn, yposIn, dx, dy, xpos, ypos, hor, ver, radius, radiusIn);

  return std::min({time1, time2, time3});
}