#include "ninja.hpp"
#include <cmath>

Ninja::Ninja()
{
  // Initialize bone structure relative positions
  bones = {{
      {0.0f, 0.0f},     // 0: Center
      {0.5f, 0.5f},     // 1: Upper right
      {0.0f, 0.7f},     // 2: Top
      {-0.5f, 0.5f},    // 3: Upper left
      {-0.7f, 0.0f},    // 4: Left
      {-0.5f, -0.5f},   // 5: Lower left
      {0.0f, -0.7f},    // 6: Bottom
      {0.5f, -0.5f},    // 7: Lower right
      {0.7f, 0.0f},     // 8: Right
      {0.35f, 0.35f},   // 9: Inner upper right
      {-0.35f, 0.35f},  // 10: Inner upper left
      {-0.35f, -0.35f}, // 11: Inner lower left
      {0.35f, -0.35f}   // 12: Inner lower right
  }};
  bonesOld = bones;
}

void Ninja::integrate()
{
  xspeed *= appliedDrag;
  yspeed *= appliedDrag;
  yspeed += appliedGravity;
  xposOld = xpos;
  yposOld = ypos;
  xpos += xspeed;
  ypos += yspeed;
}

void Ninja::preCollision()
{
  xspeedOld = xspeed;
  yspeedOld = yspeed;
  floorCount = 0;
  wallCount = 0;
  ceilingCount = 0;
  floorNormalX = 0.0f;
  floorNormalY = 0.0f;
  ceilingNormalX = 0.0f;
  ceilingNormalY = 0.0f;
  isCrushable = false;
  xCrush = 0.0f;
  yCrush = 0.0f;
  crushLen = 0.0f;
}

bool Ninja::isValidTarget() const
{
  return !(state == 6 || state == 8 || state == 9);
}

void Ninja::log(int frame)
{
  posLog.emplace_back(frame, xpos, ypos);
  speedLog.emplace_back(frame, xspeed, yspeed);
  xposLog.push_back(xpos);
  yposLog.push_back(ypos);
}

bool Ninja::hasWon() const
{
  return state == 8;
}

bool Ninja::hasDied() const
{
  return state == 6 || state == 7;
}

void Ninja::win()
{
  if (state < 6)
  {
    if (state == 3)
    {
      appliedGravity = GRAVITY_FALL;
    }
    state = 8;
  }
}

void Ninja::kill(int type, float killXpos, float killYpos, float killXspeed, float killYspeed)
{
  if (state < 6)
  {
    deathXpos = killXpos;
    deathYpos = killYpos;
    deathXspeed = killXspeed;
    deathYspeed = killYspeed;
    if (state == 3)
    {
      appliedGravity = GRAVITY_FALL;
    }
    state = 7;
  }
}

void Ninja::thinkAwaitingDeath()
{
  state = 6;
  // Note: Ragdoll implementation is omitted as it's not implemented in the original code
}

void Ninja::floorJump()
{
  jumpBuffer = -1;
  floorBuffer = -1;
  launchPadBuffer = -1;
  state = 3;
  appliedGravity = GRAVITY_JUMP;

  float jx = 0.0f;
  float jy = 0.0f;

  if (floorNormalizedX == 0.0f)
  { // Jump from flat ground
    jx = 0.0f;
    jy = -2.0f;
  }
  else
  { // Slope jump
    float dx = floorNormalizedX;
    float dy = floorNormalizedY;
    if (xspeed * dx >= 0.0f)
    { // Moving downhill
      if (xspeed * horInput >= 0.0f)
      {
        jx = 2.0f / 3.0f * dx;
        jy = 2.0f * dy;
      }
      else
      {
        jx = 0.0f;
        jy = -1.4f;
      }
    }
    else
    { // Moving uphill
      if (xspeed * horInput > 0.0f)
      { // Forward jump
        jx = 0.0f;
        jy = -1.4f;
      }
      else
      { // Perp jump
        xspeed = 0.0f;
        jx = 2.0f / 3.0f * dx;
        jy = 2.0f * dy;
      }
    }
  }

  if (yspeed > 0.0f)
  {
    yspeed = 0.0f;
  }
  xspeed += jx;
  yspeed += jy;
  xpos += jx;
  ypos += jy;
  jumpDuration = 0;
}

void Ninja::wallJump()
{
  float jx, jy;
  if (horInput * wallNormal < 0.0f && state == 5)
  { // Slide wall jump
    jx = 2.0f / 3.0f;
    jy = -1.0f;
  }
  else
  { // Regular wall jump
    jx = 1.0f;
    jy = -1.4f;
  }

  state = 3;
  appliedGravity = GRAVITY_JUMP;

  if (xspeed * wallNormal < 0.0f)
  {
    xspeed = 0.0f;
  }
  if (yspeed > 0.0f)
  {
    yspeed = 0.0f;
  }

  xspeed += jx * wallNormal;
  yspeed += jy;
  xpos += jx * wallNormal;
  ypos += jy;
  jumpBuffer = -1;
  wallBuffer = -1;
  launchPadBuffer = -1;
  jumpDuration = 0;
}

void Ninja::lpJump()
{
  floorBuffer = -1;
  wallBuffer = -1;
  jumpBuffer = -1;
  launchPadBuffer = -1;

  float boostScalar = 2.0f * std::abs(xlpBoostNormalized) + 2.0f;
  if (boostScalar == 2.0f)
  {
    boostScalar = 1.7f; // This was really needed. Thanks Metanet
  }

  xspeed += xlpBoostNormalized * boostScalar * 2.0f / 3.0f;
  yspeed += ylpBoostNormalized * boostScalar * 2.0f / 3.0f;
}

void Ninja::think()
{
  // Logic to determine if you're starting a new jump
  bool newJumpCheck = jumpInput && (jumpInputOld == 0);
  jumpInputOld = jumpInput;

  // Determine if within buffer ranges. If so, increment buffers.
  if (launchPadBuffer > -1 && launchPadBuffer < 3)
  {
    launchPadBuffer++;
  }
  else
  {
    launchPadBuffer = -1;
  }
  bool inLpBuffer = launchPadBuffer > -1 && launchPadBuffer < 4;

  if (jumpBuffer > -1 && jumpBuffer < 5)
  {
    jumpBuffer++;
  }
  else
  {
    jumpBuffer = -1;
  }
  bool inJumpBuffer = jumpBuffer > -1 && jumpBuffer < 5;

  if (wallBuffer > -1 && wallBuffer < 5)
  {
    wallBuffer++;
  }
  else
  {
    wallBuffer = -1;
  }
  bool inWallBuffer = wallBuffer > -1 && wallBuffer < 5;

  if (floorBuffer > -1 && floorBuffer < 5)
  {
    floorBuffer++;
  }
  else
  {
    floorBuffer = -1;
  }
  bool inFloorBuffer = floorBuffer > -1 && floorBuffer < 5;

  // Initiate jump buffer if beginning a new jump and airborn
  if (newJumpCheck && airborn)
  {
    jumpBuffer = 0;
  }
  // Initiate wall buffer if touched a wall this frame
  if (walled)
  {
    wallBuffer = 0;
  }
  // Initiate floor buffer if touched a floor this frame
  if (!airborn)
  {
    floorBuffer = 0;
  }

  // This part deals with the special states: dead, awaiting death, celebrating, disabled
  if (state == 6 || state == 9)
  {
    return;
  }
  if (state == 7)
  {
    thinkAwaitingDeath();
    return;
  }
  if (state == 8)
  {
    appliedDrag = airborn ? DRAG_REGULAR : DRAG_SLOW;
    return;
  }

  // This block deals with the case where the ninja is touching a floor
  if (!airborn)
  {
    float xspeedNew = xspeed + GROUND_ACCEL * horInput;
    if (std::abs(xspeedNew) < MAX_HOR_SPEED)
    {
      xspeed = xspeedNew;
    }
    if (state > 2)
    {
      if (xspeed * horInput <= 0.0f)
      {
        if (state == 3)
        {
          appliedGravity = GRAVITY_FALL;
        }
        state = 2;
      }
      else
      {
        if (state == 3)
        {
          appliedGravity = GRAVITY_FALL;
        }
        state = 1;
      }
    }
    if (!inJumpBuffer && !newJumpCheck)
    { // if not jumping
      if (state == 2)
      {
        float projection = std::abs(yspeed * floorNormalizedX - xspeed * floorNormalizedY);
        if (horInput * projection * xspeed > 0.0f)
        {
          state = 1;
          return;
        }
        if (projection < 0.1f && floorNormalizedX == 0.0f)
        {
          state = 0;
          return;
        }
        if (yspeed < 0.0f && floorNormalizedX != 0.0f)
        {
          // Up slope friction formula
          float speedScalar = std::sqrt(xspeed * xspeed + yspeed * yspeed);
          float fricForce = std::abs(xspeed * (1.0f - FRICTION_GROUND) * floorNormalizedY);
          float fricForce2 = speedScalar - fricForce * floorNormalizedY * floorNormalizedY;
          xspeed = xspeed / speedScalar * fricForce2;
          yspeed = yspeed / speedScalar * fricForce2;
          return;
        }
        xspeed *= FRICTION_GROUND;
        return;
      }
      if (state == 1)
      {
        float projection = std::abs(yspeed * floorNormalizedX - xspeed * floorNormalizedY);
        if (horInput * projection * xspeed > 0.0f)
        {
          // if holding inputs in downhill direction or flat ground
          if (horInput * floorNormalizedX >= 0.0f)
          {
            return;
          }
          if (std::abs(xspeedNew) < MAX_HOR_SPEED)
          {
            float boost = GROUND_ACCEL / 2.0f * horInput;
            float xboost = boost * floorNormalizedY * floorNormalizedY;
            float yboost = boost * floorNormalizedY * -floorNormalizedX;
            xspeed += xboost;
            yspeed += yboost;
          }
          return;
        }
        state = 2;
      }
      else
      { // if you were in state 0
        if (horInput != 0.0f)
        {
          state = 1;
          return;
        }
        float projection = std::abs(yspeed * floorNormalizedX - xspeed * floorNormalizedY);
        if (projection < 0.1f)
        {
          xspeed *= FRICTION_GROUND_SLOW;
          return;
        }
        state = 2;
      }
      return;
    }
    floorJump(); // if you're jumping
    return;
  }

  // This block deals with the case where the ninja didn't touch a floor
  else
  {
    float xspeedNew = xspeed + AIR_ACCEL * horInput;
    if (std::abs(xspeedNew) < MAX_HOR_SPEED)
    {
      xspeed = xspeedNew;
    }
    if (state < 3)
    {
      state = 4;
      return;
    }
    if (state == 3)
    {
      jumpDuration++;
      if (!jumpInput || jumpDuration > MAX_JUMP_DURATION)
      {
        appliedGravity = GRAVITY_FALL;
        state = 4;
        return;
      }
    }
    if (inJumpBuffer || newJumpCheck)
    { // if able to perform jump
      if (walled || inWallBuffer)
      {
        wallJump();
        return;
      }
      if (inFloorBuffer)
      {
        floorJump();
        return;
      }
      if (inLpBuffer && newJumpCheck)
      {
        lpJump();
        return;
      }
    }
    if (!walled)
    {
      if (state == 5)
      {
        state = 4;
      }
    }
    else
    {
      if (state == 5)
      {
        if (horInput * wallNormal <= 0.0f)
        {
          yspeed *= FRICTION_WALL;
        }
        else
        {
          state = 4;
        }
      }
      else
      {
        if (yspeed > 0.0f && horInput * wallNormal < 0.0f)
        {
          if (state == 3)
          {
            appliedGravity = GRAVITY_FALL;
          }
          state = 5;
        }
      }
    }
  }
}

void Ninja::updateGraphics()
{
  int animStateOld = animState;

  if (state == 5)
  {
    animState = 4;
    tilt = 0.0f;
    facing = -wallNormal;
    animRate = yspeed;
  }
  else if (!airborn && state != 3)
  {
    tilt = std::atan2(floorNormalizedY, floorNormalizedX) + M_PI / 2.0f;
    if (state == 0)
    {
      animState = 0;
    }
    if (state == 1)
    {
      animState = 1;
      animRate = std::abs(yspeed * floorNormalizedX - xspeed * floorNormalizedY);
      if (horInput != 0.0f)
      {
        facing = horInput > 0.0f ? 1 : -1;
      }
    }
    if (state == 2)
    {
      animState = 2;
      animRate = std::abs(yspeed * floorNormalizedX - xspeed * floorNormalizedY);
    }
    if (state == 8)
    {
      animState = 6;
    }
  }
  else
  {
    animState = 3;
    animRate = yspeed;
    if (state == 3)
    {
      tilt = 0.0f;
    }
    else
    {
      tilt *= 0.9f;
    }
  }

  if (state != 5)
  {
    if (std::abs(xspeed) > 0.01f)
    {
      facing = xspeed > 0.0f ? 1 : -1;
    }
  }

  if (animState != animStateOld)
  {
    if (animState == 0)
    {
      if (animFrame > 0)
      {
        animFrame = 1;
      }
    }
    if (animState == 1)
    {
      if (animStateOld != 3)
      {
        if (animStateOld == 2)
        {
          animFrame = 39;
          runCycle = 162;
          frameResidual = 0.0f;
        }
        else
        {
          animFrame = 12;
          runCycle = 0;
          frameResidual = 0.0f;
        }
      }
      else
      {
        animFrame = 18;
        runCycle = 36;
        frameResidual = 0.0f;
      }
    }
    if (animState == 2)
    {
      animFrame = 0;
    }
    if (animState == 3)
    {
      animFrame = 84;
    }
    if (animState == 4)
    {
      animFrame = 103;
    }
    if (animState == 6)
    {
      // Note: Dance functionality is omitted for now as it requires random number generation
      // and dance dictionary which wasn't provided in the original C++ code
      animFrame = 0; // Default dance frame
    }
  }

  if (animState == 0)
  {
    if (animFrame < 11)
    {
      animFrame++;
    }
  }
  if (animState == 1)
  {
    float newCycle = animRate / 0.15f + frameResidual;
    frameResidual = newCycle - std::floor(newCycle);
    runCycle = (runCycle + static_cast<int>(std::floor(newCycle))) % 432;
    animFrame = runCycle / 6 + 12;
  }
  if (animState == 3)
  {
    float rate;
    if (animRate >= 0.0f)
    {
      rate = std::sqrt(std::min(animRate * 0.6f, 1.0f));
    }
    else
    {
      rate = std::max(animRate * 1.5f, -1.0f);
    }
    animFrame = 93 + static_cast<int>(std::floor(9.0f * rate));
  }

  // Note: Dance animation update is omitted as it requires dance dictionary

  bonesOld = bones;
  // Note: calcNinjaPosition() is omitted as it requires animation data
}