#include "segment.hpp"
#include <cmath>

Segment::Segment(float x1_, float y1_, float x2_, float y2_)
    : x1(x1_), y1(y1_), x2(x2_), y2(y2_)
{
  float dx = x2 - x1;
  float dy = y2 - y1;
  length = std::sqrt(dx * dx + dy * dy);
  if (length > 0)
  {
    nx = -dy / length; // Normal points to the left of the segment
    ny = dx / length;
  }
  else
  {
    nx = 0;
    ny = 1;
  }
}

float Segment::intersectWithRay(float xpos, float ypos, float dx, float dy, float radius) const
{
  if (!active)
    return 1.0f;

  float wx = x2 - x1;
  float wy = y2 - y1;

  if (length == 0)
    return 1.0f;

  // Project ray direction onto segment normal
  float projVel = dx * nx + dy * ny;
  if (projVel >= 0)
    return 1.0f; // Ray moving away from segment

  // Project point-to-segment vector onto normal
  float projPos = (xpos - x1) * nx + (ypos - y1) * ny - radius;
  if (projPos < 0)
    return 0.0f; // Already penetrating

  float t = projPos / -projVel;
  if (t > 1)
    return 1.0f; // Intersection too far

  // Check if intersection point lies within segment bounds
  float ix = xpos + t * dx;
  float iy = ypos + t * dy;
  float projSeg = (ix - x1) * wx + (iy - y1) * wy;

  return (0 <= projSeg && projSeg <= length * length) ? t : 1.0f;
}

std::tuple<bool, float, float> Segment::getClosestPoint(float xpos, float ypos) const
{
  if (!active)
    return {false, 0.0f, 0.0f};

  float wx = x2 - x1;
  float wy = y2 - y1;
  float dx = xpos - x1;
  float dy = ypos - y1;

  // Project point onto segment
  float proj = (dx * wx + dy * wy) / (length * length);
  proj = std::max(0.0f, std::min(1.0f, proj));

  // Calculate closest point
  float cx = x1 + proj * wx;
  float cy = y1 + proj * wy;

  // Check which side of the segment the point is on
  bool isBackFacing = (dx * nx + dy * ny) < 0;

  return {isBackFacing, cx, cy};
}