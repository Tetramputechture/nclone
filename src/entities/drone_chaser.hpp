#pragma once

#include "drone_zap.hpp"

class DroneChaser : public DroneZap
{
public:
  static constexpr int ENTITY_TYPE = 15;
  static constexpr int MAX_COUNT_PER_LEVEL = 256;

  DroneChaser(Simulation *sim, float xcoord, float ycoord, int orientation, int mode);

  void think() override;

protected:
  void chooseNextDirectionAndGoal() override;
};