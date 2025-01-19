#pragma once

class EntityCollisionResult
{
public:
  // For physical collisions
  float depenetrationX = 0.0f;
  float depenetrationY = 0.0f;
  float normalX = 0.0f;
  float normalY = 0.0f;
  bool hasCollision = false;

  EntityCollisionResult(float depX = 0.0f, float depY = 0.0f, float normX = 0.0f, float normY = 0.0f, bool collision = false)
      : depenetrationX(depX), depenetrationY(depY), normalX(normX), normalY(normY), hasCollision(collision)
  {
  }

  static EntityCollisionResult noCollision()
  {
    return EntityCollisionResult();
  }

  operator bool() const { return hasCollision; }
};
