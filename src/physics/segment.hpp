#pragma once

#include <tuple>

class Segment
{
public:
  Segment(float x1, float y1, float x2, float y2);

  // Intersection tests
  float intersectWithRay(float xpos, float ypos, float dx, float dy, float radius) const;
  std::tuple<bool, float, float> getClosestPoint(float xpos, float ypos) const;

  // Getters
  float getX1() const { return x1; }
  float getY1() const { return y1; }
  float getX2() const { return x2; }
  float getY2() const { return y2; }

  // State
  bool active = true;

private:
  float x1, y1, x2, y2; // Endpoint coordinates
  float nx, ny;         // Normal vector
  float length;         // Segment length
};