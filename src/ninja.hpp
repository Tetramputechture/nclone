#pragma once

#include <array>
#include <utility>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <unordered_map>
#include "physics/physics.hpp"

class Simulation;

class Ninja
{
public:
  // Physics constants
  static constexpr float GRAVITY_FALL = 0.06666666666666665f;
  static constexpr float GRAVITY_JUMP = 0.01111111111111111f;
  static constexpr float GROUND_ACCEL = 0.06666666666666665f;
  static constexpr float AIR_ACCEL = 0.04444444444444444f;
  static constexpr float DRAG_REGULAR = 0.9933221725495059f;         // 0.99^(2/3)
  static constexpr float DRAG_SLOW = 0.8617738760127536f;            // 0.80^(2/3)
  static constexpr float FRICTION_GROUND = 0.9459290248857720f;      // 0.92^(2/3)
  static constexpr float FRICTION_GROUND_SLOW = 0.8617738760127536f; // 0.80^(2/3)
  static constexpr float FRICTION_WALL = 0.9113380468927672f;        // 0.87^(2/3)
  static constexpr float MAX_HOR_SPEED = 3.333333333333333f;
  static constexpr int MAX_JUMP_DURATION = 45;
  static constexpr float MAX_SURVIVABLE_IMPACT = 6.0f;
  static constexpr float MIN_SURVIVABLE_CRUSHING = 0.05f;
  static constexpr float RADIUS = 10.0f;

  // Position and velocity
  float xpos = 0.0f;
  float ypos = 0.0f;
  float xspeed = 0.0f;
  float yspeed = 0.0f;
  float xposOld = 0.0f;
  float yposOld = 0.0f;
  float xspeedOld = 0.0f;
  float yspeedOld = 0.0f;
  float deathXpos = 0.0f;
  float deathYpos = 0.0f;
  float deathXspeed = 0.0f;
  float deathYspeed = 0.0f;

  // Physics state
  float appliedGravity = GRAVITY_FALL;
  float appliedDrag = DRAG_REGULAR;
  float appliedFriction = FRICTION_GROUND;
  int state = 0; // 0:Immobile, 1:Running, 2:Ground sliding, 3:Jumping, 4:Falling, 5:Wall sliding
  bool airborn = false;
  bool airbornOld = false;
  bool walled = false;

  // Jump state
  int jumpInputOld = 0;
  int jumpDuration = 0;
  int jumpBuffer = -1;
  int floorBuffer = -1;
  int wallBuffer = -1;
  int launchPadBuffer = -1;

  // Floor and wall normals
  float floorNormalizedX = 0.0f;
  float floorNormalizedY = -1.0f;
  float ceilingNormalizedX = 0.0f;
  float ceilingNormalizedY = 1.0f;
  float wallNormal = 0.0f;
  float xlpBoostNormalized = 0.0f;
  float ylpBoostNormalized = 0.0f;

  // Animation state
  int animState = 0;
  int facing = 1;
  float tilt = 0.0f;
  float animRate = 0.0f;
  int animFrame = 11;
  float frameResidual = 0.0f;
  int runCycle = 0;
  int danceId = 0;

  // Collision state
  int floorCount = 0;
  int ceilingCount = 0;
  float floorNormalX = 0.0f;
  float floorNormalY = 0.0f;
  float ceilingNormalX = 0.0f;
  float ceilingNormalY = 0.0f;
  bool isCrushable = false;
  float xCrush = 0.0f;
  float yCrush = 0.0f;
  float crushLen = 0.0f;

  // Game state
  int goldCollected = 0;
  int doorsOpened = 0;

  // Input state
  float horInput = 0.0f;
  int jumpInput = 0;

  // Simulation reference
  Simulation *sim = nullptr;

  // Entity type for collision handling
  int entityType = 0;

  // Bone structure
  static constexpr int NUM_BONES = 13;
  std::array<std::pair<float, float>, NUM_BONES> bones;
  std::array<std::pair<float, float>, NUM_BONES> bonesOld;

  // Position and speed logs
  std::vector<std::tuple<int, float, float>> posLog;
  std::vector<std::tuple<int, float, float>> speedLog;
  std::vector<float> xposLog;
  std::vector<float> yposLog;

  // Animation data
  static constexpr const char *ANIM_DATA_FILE = "anim_data_line_new.txt.bin";
  static std::vector<std::array<std::pair<float, float>, 13>> cachedNinjaAnimation;
  static void loadNinjaAnimation();
  bool ninjaAnimMode = false;
  std::vector<std::array<std::pair<float, float>, 13>> ninjaAnimation;
  void calcNinjaPosition();

  // Dance parameters
  static constexpr bool DANCE_RANDOM = true;
  static constexpr int DANCE_ID_DEFAULT = 0;
  struct DanceRange
  {
    int start;
    int end;
  };
  static const std::unordered_map<int, DanceRange> DANCE_DIC;

  // Methods
  Ninja();
  explicit Ninja(Simulation *simulation);
  void integrate();
  void preCollision();
  void collideVsObjects();
  void collideVsTiles();
  void postCollision();
  void floorJump();
  void wallJump();
  void lpJump();
  void think();
  void thinkAwaitingDeath();
  void updateGraphics();
  void win();
  void kill(int type, float xpos, float ypos, float xspeed, float yspeed);
  bool isValidTarget() const;
  void log(int frame);
  bool hasWon() const;
  bool hasDied() const;

  // Getters and setters
  void setHorInput(float input) { horInput = input; }
  void setJumpInput(int input) { jumpInput = input; }
  void setAnimFrame(int frame) { animFrame = frame; }
  void setAnimState(int state) { animState = state; }
  int getState() const { return state; }

private:
  void initializeBones();
};