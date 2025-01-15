#pragma once

#include "segment.hpp"
#include <utility>
#include <cmath>

class GridSegmentCircular : public Segment
{
public:
  GridSegmentCircular(std::pair<float, float> center, std::pair<int, int> quadrant, bool convex, float radius = 24.0f);

  std::tuple<bool, float, float> getClosestPoint(float xpos, float ypos) const override;
  float intersectWithRay(float xpos, float ypos, float dx, float dy, float radius) const override;

  // Implement virtual getters
  const char *getType() const override { return type; }
  float getRadius() const override { return radius; }
  float getStartAngle() const override { return std::atan2(ver, hor); }
  float getEndAngle() const override { return std::atan2(ver, hor) + M_PI / 2; }

private:
  float xpos, ypos;             // Center position
  int hor, ver;                 // Quadrant direction
  float radius;                 // Radius of the quarter-circle
  std::pair<float, float> pHor; // Horizontal extremity
  std::pair<float, float> pVer; // Vertical extremity
  bool active;                  // Whether the segment is currently active
  const char *type = "circular";
  bool convex; // Whether the quarter-circle is convex or concave
};