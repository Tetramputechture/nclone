#include "ninja.hpp"
#include "simulation.hpp"
#include "physics/physics.hpp"
#include "entities/entity.hpp"
#include <filesystem>
#include <unordered_map>
#include <random>

// Initialize static members
std::vector<std::array<std::pair<float, float>, 13>> Ninja::cachedNinjaAnimation;
const std::unordered_map<int, Ninja::DanceRange> Ninja::DANCE_DIC = {
    {0, {104, 104}}, {1, {106, 225}}, {2, {226, 345}}, {3, {346, 465}}, {4, {466, 585}}, {5, {586, 705}}, {6, {706, 825}}, {7, {826, 945}}, {8, {946, 1065}}, {9, {1066, 1185}}, {10, {1186, 1305}}, {11, {1306, 1485}}, {12, {1486, 1605}}, {13, {1606, 1664}}, {14, {1665, 1731}}, {15, {1732, 1810}}, {16, {1811, 1852}}, {17, {1853, 1946}}, {18, {1947, 2004}}, {19, {2005, 2156}}, {20, {2157, 2241}}, {21, {2242, 2295}}};

void Ninja::loadNinjaAnimation()
{
  if (!cachedNinjaAnimation.empty())
  {
    return;
  }

  std::ifstream file(ANIM_DATA_FILE, std::ios::binary);
  if (!file)
  {
    return;
  }

  uint32_t frames;
  file.read(reinterpret_cast<char *>(&frames), sizeof(frames));

  cachedNinjaAnimation.resize(frames);
  for (uint32_t i = 0; i < frames; ++i)
  {
    for (int j = 0; j < 13; ++j)
    {
      double x, y;
      file.read(reinterpret_cast<char *>(&x), sizeof(x));
      file.read(reinterpret_cast<char *>(&y), sizeof(y));
      cachedNinjaAnimation[i][j] = {static_cast<float>(x), static_cast<float>(y)};
    }
  }
}

Ninja::Ninja(float xPos, float yPos)
{
  xpos = xPos;
  ypos = yPos;
  xposOld = xPos;
  yposOld = yPos;
  initializeBones();
  ninjaAnimMode = std::filesystem::exists(ANIM_DATA_FILE);
  if (ninjaAnimMode)
  {
    loadNinjaAnimation();
    ninjaAnimation = cachedNinjaAnimation;
  }
}

void Ninja::initializeBones()
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
      if (horInput != 0)
      {
        facing = horInput > 0 ? 1 : -1;
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
      // Choose dance animation
      if (DANCE_RANDOM)
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, DANCE_DIC.size() - 1);
        auto it = DANCE_DIC.begin();
        std::advance(it, dis(gen));
        danceId = it->first;
      }
      else
      {
        danceId = DANCE_ID_DEFAULT;
      }
      animFrame = DANCE_DIC.at(danceId).start;
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
  if (animState == 6)
  {
    if (animFrame < DANCE_DIC.at(danceId).end)
    {
      animFrame++;
    }
  }

  bonesOld = bones;
  if (ninjaAnimMode)
  {
    calcNinjaPosition();
  }
}

void Ninja::collideVsObjects(Simulation &sim)
{
  // Get entities from neighborhood
  auto entities = Physics::gatherEntitiesFromNeighbourhood(sim, xpos, ypos);

  for (auto *entity : entities)
  {
    if (entity->isPhysicalCollidable())
    {
      auto collision = entity->physicalCollision();
      if (collision)
      {
        // Get depenetration values from collision result
        // r1 = normal.x, r2 = normal.y, r3 = depenetration length
        float normalX = collision->getR1();
        float normalY = collision->getR2().value_or(0.0f);
        float depenLen = collision->getR3().value_or(0.0f);

        // Calculate depenetration vector
        float popX = normalX * depenLen;
        float popY = normalY * depenLen;
        xpos += popX;
        ypos += popY;

        // Update crushing parameters unless collision with bounce block
        if (entity->getType() != 17)
        {
          xCrush += popX;
          yCrush += popY;
          crushLen += depenLen;
        }

        // Ninja can only get crushed if collision with thwump
        if (entity->getType() == 20)
        {
          isCrushable = true;
        }

        // Depenetration for bounce blocks, thwumps and shwumps
        if (entity->getType() == 17 || entity->getType() == 20 || entity->getType() == 28)
        {
          xspeed += popX;
          yspeed += popY;
        }

        // Handle one-way platform collisions
        if (entity->getType() == 11)
        {
          float xspeedNew = (xspeed * normalY - yspeed * normalX) * normalY;
          float yspeedNew = (xspeed * normalY - yspeed * normalX) * (-normalX);
          xspeed = xspeedNew;
          yspeed = yspeedNew;
        }

        // Adjust ceiling variables if ninja collides with ceiling (or wall!)
        if (normalY >= -0.0001f)
        {
          ceilingCount++;
          ceilingNormalX += normalX;
          ceilingNormalY += normalY;
        }
        else // Adjust floor variables if ninja collides with floor
        {
          floorCount++;
          floorNormalX += normalX;
          floorNormalY += normalY;
        }
      }
    }
  }
}

void Ninja::collideVsTiles(Simulation &sim)
{
  // Interpolation routine mainly to prevent from going through walls
  float dx = xpos - xposOld;
  float dy = ypos - yposOld;
  float time = Physics::sweepCircleVsTiles(sim, xposOld, yposOld, dx, dy, RADIUS * 0.5f);
  xpos = xposOld + time * dx;
  ypos = yposOld + time * dy;

  // Find the closest point from the ninja, apply depenetration and update speed. Loop 32 times.
  for (int i = 0; i < 32; i++)
  {
    auto result = Physics::getSingleClosestPoint(sim, xpos, ypos, RADIUS);
    if (!result)
      break;

    const auto &[isBackFacing, closestPoint] = *result;
    const auto &[a, b] = closestPoint;

    float dx = xpos - a;
    float dy = ypos - b;

    // Handle corner cases
    if (std::abs(dx) <= 0.0000001f)
    {
      dx = 0;
      if (xpos == 50.51197510492316f || xpos == 49.23232124849253f)
      {
        dx = -std::pow(2.0f, -47.0f);
      }
      if (xpos == 49.153536108584795f)
      {
        dx = std::pow(2.0f, -47.0f);
      }
    }

    float dist = std::sqrt(dx * dx + dy * dy);
    float depenLen = RADIUS - dist * (isBackFacing ? -1.0f : 1.0f);

    if (dist == 0 || depenLen < 0.0000001f)
      return;

    float depenX = dx / dist * depenLen;
    float depenY = dy / dist * depenLen;
    xpos += depenX;
    ypos += depenY;
    xCrush += depenX;
    yCrush += depenY;
    crushLen += depenLen;

    float dotProduct = xspeed * dx + yspeed * dy;
    if (dotProduct < 0) // Project velocity onto surface only if moving towards surface
    {
      float xspeedNew = (xspeed * dy - yspeed * dx) / (dist * dist) * dy;
      float yspeedNew = (xspeed * dy - yspeed * dx) / (dist * dist) * (-dx);
      xspeed = xspeedNew;
      yspeed = yspeedNew;
    }

    // Adjust ceiling variables if ninja collides with ceiling (or wall!)
    if (dy >= -0.0001f)
    {
      ceilingCount++;
      ceilingNormalX += dx / dist;
      ceilingNormalY += dy / dist;
    }
    else // Adjust floor variables if ninja collides with floor
    {
      floorCount++;
      floorNormalX += dx / dist;
      floorNormalY += dy / dist;
    }
  }
}

void Ninja::calcNinjaPosition()
{
  if (!ninjaAnimMode || ninjaAnimation.empty())
  {
    return;
  }

  // Create temporary bones array for new positions
  std::array<std::pair<float, float>, NUM_BONES> newBones;

  // Get bones from animation frame
  const auto &animFrameBones = ninjaAnimation[animFrame];
  for (int i = 0; i < NUM_BONES; ++i)
  {
    newBones[i] = animFrameBones[i];
  }

  // Handle running animation interpolation
  if (animState == 1)
  {
    float interpolation = static_cast<float>(runCycle % 6) / 6.0f;
    if (interpolation > 0)
    {
      const auto &nextBones = ninjaAnimation[(animFrame - 12) % 72 + 12];
      for (int i = 0; i < NUM_BONES; ++i)
      {
        newBones[i].first += interpolation * (nextBones[i].first - newBones[i].first);
        newBones[i].second += interpolation * (nextBones[i].second - newBones[i].second);
      }
    }
  }

  // Apply facing direction and tilt
  for (int i = 0; i < NUM_BONES; ++i)
  {
    newBones[i].first *= facing;
    float x = newBones[i].first;
    float y = newBones[i].second;
    float tcos = std::cos(tilt);
    float tsin = std::sin(tilt);
    newBones[i].first = x * tcos - y * tsin;
    newBones[i].second = x * tsin + y * tcos;
  }

  // Swap bone arrays
  bonesOld = bones;
  bones = newBones;
}

void Ninja::postCollision(Simulation &sim)
{
  // Perform LOGICAL collisions between the ninja and nearby entities.
  // Also check if the ninja can interact with the walls of entities when applicable.
  float wallNormalSum = 0.0f;
  auto entities = Physics::gatherEntitiesFromNeighbourhood(sim, xpos, ypos);
  for (auto *entity : entities)
  {
    if (entity->isLogicalCollidable())
    {
      auto collisionResult = entity->logicalCollision();
      if (collisionResult)
      {
        if (entity->getType() == 10)
        { // If collision with launch pad, update speed and position
          float normalX = collisionResult->getR1();
          float normalY = collisionResult->getR2().value_or(0.0f);
          float depenLen = collisionResult->getR3().value_or(0.0f);
          float xboost = normalX * depenLen * 2.0f / 3.0f;
          float yboost = normalY * depenLen * 2.0f / 3.0f;
          xpos += xboost;
          ypos += yboost;
          xspeed = xboost;
          yspeed = yboost;
          floorCount = 0;
          floorBuffer = -1;
          float boostScalar = std::sqrt(xboost * xboost + yboost * yboost);
          xlpBoostNormalized = xboost / boostScalar;
          ylpBoostNormalized = yboost / boostScalar;
          launchPadBuffer = 0;
          if (state == 3)
          {
            appliedGravity = GRAVITY_FALL;
          }
          state = 4;
        }
        else
        { // If touched wall of bounce block, oneway, thwump or shwump, retrieve wall normal
          wallNormalSum += collisionResult->getR1();
        }
      }
    }
  }

  // Check if the ninja can interact with walls from nearby tile segments
  float rad = RADIUS + 0.1f;
  auto segments = sim.getSegmentsInRegion(xpos - rad, ypos - rad, xpos + rad, ypos + rad);
  for (const auto &segment : segments)
  {
    bool valid;
    float a, b;
    std::tie(valid, a, b) = segment->getClosestPoint(xpos, ypos);
    if (!valid)
      continue;

    float dx = xpos - a;
    float dy = ypos - b;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (std::abs(dy) < 0.00001f && dist > 0.0f && dist <= rad)
    {
      wallNormalSum += dx / dist;
    }
  }

  // Check if airborn or walled
  airbornOld = airborn;
  airborn = true;
  walled = false;
  if (wallNormalSum != 0.0f)
  {
    walled = true;
    wallNormal = wallNormalSum / std::abs(wallNormalSum);
  }

  // Calculate the combined floor normalized normal vector if the ninja has touched any floor
  if (floorCount > 0)
  {
    airborn = false;
    float floorScalar = std::sqrt(floorNormalX * floorNormalX + floorNormalY * floorNormalY);
    if (floorScalar == 0.0f)
    {
      floorNormalizedX = 0.0f;
      floorNormalizedY = -1.0f;
    }
    else
    {
      floorNormalizedX = floorNormalX / floorScalar;
      floorNormalizedY = floorNormalY / floorScalar;
    }
    if (state != 8 && airbornOld)
    { // Check if died from floor impact
      float impactVel = -(floorNormalizedX * xspeedOld + floorNormalizedY * yspeedOld);
      if (impactVel > MAX_SURVIVABLE_IMPACT - 4.0f / 3.0f * std::abs(floorNormalizedY))
      {
        xspeed = xspeedOld;
        yspeed = yspeedOld;
        kill(1, xpos, ypos, xspeed * 0.5f, yspeed * 0.5f);
      }
    }
  }

  // Calculate the combined ceiling normalized normal vector if the ninja has touched any ceiling
  if (ceilingCount > 0)
  {
    float ceilingScalar = std::sqrt(ceilingNormalX * ceilingNormalX + ceilingNormalY * ceilingNormalY);
    if (ceilingScalar == 0.0f)
    {
      ceilingNormalizedX = 0.0f;
      ceilingNormalizedY = 1.0f;
    }
    else
    {
      ceilingNormalizedX = ceilingNormalX / ceilingScalar;
      ceilingNormalizedY = ceilingNormalY / ceilingScalar;
    }
    if (state != 8)
    { // Check if died from ceiling impact
      float impactVel = -(ceilingNormalizedX * xspeedOld + ceilingNormalizedY * yspeedOld);
      if (impactVel > MAX_SURVIVABLE_IMPACT - 4.0f / 3.0f * std::abs(ceilingNormalizedY))
      {
        xspeed = xspeedOld;
        yspeed = yspeedOld;
        kill(1, xpos, ypos, xspeed * 0.5f, yspeed * 0.5f);
      }
    }
  }

  // Check if ninja died from crushing
  if (isCrushable && crushLen > 0.0f)
  {
    if (std::sqrt(xCrush * xCrush + yCrush * yCrush) / crushLen < MIN_SURVIVABLE_CRUSHING)
    {
      kill(2, xpos, ypos, 0.0f, 0.0f);
    }
  }
}