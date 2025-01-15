#pragma once

#include "../physics/segment.hpp"
#include <utility>

class GridSegmentLinear : public Segment
{
public:
  GridSegmentLinear(std::pair<float, float> p1, std::pair<float, float> p2, bool oriented = true);

  /**
   * Find the closest point on the segment from the given position.
   * isBackFacing is false if the position is facing the segment's outer edge.
   */
  std::tuple<bool, float, float> getClosestPoint(float xpos, float ypos) const;

  /**
   * Return the time of intersection (as a fraction of a frame) for the collision
   * between the segment and a circle moving along a given direction. Return 0 if the circle
   * is already intersecting or 1 if it won't intersect within the frame.
   */
  float intersectWithRay(float xpos, float ypos, float dx, float dy, float radius) const;

  float x1, y1;  // First endpoint
  float x2, y2;  // Second endpoint
  bool oriented; // Whether the segment has an inner/outer side
  bool active;   // Whether the segment is currently active
  const char *type = "linear";
};