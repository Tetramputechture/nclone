#pragma once

#include <memory>
#include <vector>
#include "simulation.hpp"
#include "renderer.hpp"
#include "ninja.hpp"

class SimWrapper
{
public:
  SimWrapper(bool enableDebugOverlay = false, bool basicSim = false, bool fullExport = false, float tolerance = 1.0, bool enableAnim = true, bool logData = false, const std::string &renderMode = "rgb_array");
  ~SimWrapper() = default;

  // Simulation control
  void loadMap(const std::vector<uint8_t> &mapData);
  void reset();
  void tick(int horInput, int jumpInput);

  // State getters
  bool hasWon() const;
  bool hasDied() const;
  std::pair<float, float> getNinjaPosition() const;
  std::pair<float, float> getNinjaVelocity() const;
  bool isNinjaInAir() const;
  bool isNinjaWalled() const;
  int getGoldCollected() const;
  int getDoorsOpened() const;
  int getSimFrame() const;
  int getTotalGoldAvailable() const;
  bool exitSwitchActivated() const;
  std::pair<float, float> getExitSwitchPosition() const;
  std::pair<float, float> getExitDoorPosition() const;

  // New state getters
  std::vector<float> getNinjaState() const;
  std::vector<float> getEntityStates(bool onlyExitAndSwitch = false) const;
  std::vector<float> getStateVector(bool onlyExitAndSwitch = false) const;

  // Rendering
  void render(std::vector<float> &fullBuffer, std::vector<float> &playerViewBuffer,
              int fullViewWidth = DEFAULT_FULL_VIEW_WIDTH,
              int fullViewHeight = DEFAULT_FULL_VIEW_HEIGHT,
              int playerViewWidth = DEFAULT_PLAYER_VIEW_WIDTH,
              int playerViewHeight = DEFAULT_PLAYER_VIEW_HEIGHT);

  // Window status
  bool isWindowOpen() const;

private:
  std::unique_ptr<Simulation> sim;
  std::unique_ptr<Renderer> renderer;
  SimConfig simConfig;
  std::string renderMode;
  static const int DEFAULT_FULL_VIEW_WIDTH = 176;
  static const int DEFAULT_FULL_VIEW_HEIGHT = 100;
  static const int DEFAULT_PLAYER_VIEW_WIDTH = 84;
  static const int DEFAULT_PLAYER_VIEW_HEIGHT = 84;
};
