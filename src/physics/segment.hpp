#pragma once

#include <tuple>

class Segment
{
public:
  Segment(float x1_, float y1_, float x2_, float y2_);
  virtual ~Segment() = default;

  float getX1() const { return x1; }
  float getY1() const { return y1; }
  float getX2() const { return x2; }
  float getY2() const { return y2; }
  bool isActive() const { return active; }

  // Virtual getters that derived classes must implement
  virtual const char *getType() const = 0;
  virtual float getRadius() const { return 0.0f; }
  virtual float getStartAngle() const { return 0.0f; }
  virtual float getEndAngle() const { return 0.0f; }

  virtual std::tuple<bool, float, float> getClosestPoint(float xpos, float ypos) const = 0;
  virtual float intersectWithRay(float xpos, float ypos, float dx, float dy, float radius) const = 0;

protected:
  float x1, y1;
  float x2, y2;
  float length;
  float nx, ny;
  bool active = true;
};