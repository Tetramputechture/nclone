#pragma once

#include <optional>
#include <utility>

/**
 * @brief A class that represents the result of an entity collision.
 *
 * In our original Python simulation, each Entity could have a logical_collision
 * or physical_collision method. These methods would return a tuple of floats
 * that represented the result of the collision.
 *
 * In our new C++ simulation, we want to be able to return a single object
 * that represents the result of the collision.
 *
 * This class is a way to represent the result of the collision in a way that
 * mimicks the original Python simulation, and is type safe until we want to
 * refactor our collision result objects.
 *
 */
class EntityCollisionResult
{
public:
  EntityCollisionResult(float r1val, std::optional<float> r2val, std::optional<float> r3val, std::optional<float> r4val)
      : r1(r1val), r2(r2val), r3(r3val), r4(r4val) {}

  float getR1() const { return r1; }
  std::optional<float> getR2() const { return r2; }
  std::optional<float> getR3() const { return r3; }
  std::optional<float> getR4() const { return r4; }

private:
  const float r1;
  const std::optional<float> r2;
  const std::optional<float> r3;
  const std::optional<float> r4;
};
