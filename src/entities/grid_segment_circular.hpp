#pragma once

#include "../physics/segment.hpp"
#include <utility>

class GridSegmentCircular : public Segment
{
public:
  GridSegmentCircular(std::pair<float, float> center, std::pair<int, int> quadrant, bool convex, float radius = 24.0f);

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

  float xpos, ypos;             // Center position
  int hor, ver;                 // Quadrant direction
  float radius;                 // Radius of the quarter-circle
  std::pair<float, float> pHor; // Horizontal extremity
  std::pair<float, float> pVer; // Vertical extremity
  bool active;                  // Whether the segment is currently active
  const char *type = "circular";
  bool convex; // Whether the quarter-circle is convex or concave
};