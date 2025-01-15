#pragma once

class SimConfig
{
public:
  // Constructor with default values matching Python version
  SimConfig(bool basicSim = false,
            bool fullExport = false,
            float tolerance = 1.0f,
            bool enableAnim = true,
            bool logData = false);

  // Static factory method (equivalent to from_args in Python)
  static SimConfig fromArgs(const struct SimConfigArgs *args = nullptr);

  // Configuration flags
  bool basicSim;
  bool fullExport;
  float tolerance;
  bool enableAnim;
  bool logData;
};

// Helper struct for passing arguments (similar to Python's args)
struct SimConfigArgs
{
  bool basicSim = false;
  bool fullExport = false;
  float tolerance = 1.0f;
  bool enableAnim = true;
  bool logData = false;
};