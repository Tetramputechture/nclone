#pragma once

#include "segment.hpp"
#include <utility>

class GridSegmentLinear : public Segment
{
public:
  GridSegmentLinear(std::pair<float, float> p1, std::pair<float, float> p2, bool oriented = true);

  std::tuple<bool, float, float> getClosestPoint(float xpos, float ypos) const override;
  float intersectWithRay(float xpos, float ypos, float dx, float dy, float radius) const override;

  // Implement virtual getters
  const char *getType() const override { return type; }

private:
  float x1, y1;  // First endpoint
  float x2, y2;  // Second endpoint
  bool oriented; // Whether the segment has an inner/outer side
  bool active;   // Whether the segment is currently active
  const char *type = "linear";
};