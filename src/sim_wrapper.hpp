#pragma once

#include <memory>
#include <vector>
#include "simulation.hpp"
#include "renderer.hpp"
#include "ninja.hpp"

class SimWrapper
{
public:
  SimWrapper(bool enableDebugOverlay = false, bool basicSim = false, bool fullExport = false, float tolerance = 1.0, bool enableAnim = true, bool logData = false);
  ~SimWrapper() = default;

  // Simulation control
  void loadMap(const std::vector<uint8_t> &mapData);
  void reset();
  void tick(float horInput, int jumpInput);

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
  void renderToBuffer(std::vector<float> &buffer);

private:
  std::unique_ptr<Simulation> sim;
  std::unique_ptr<Renderer> renderer;
  SimConfig simConfig;
};