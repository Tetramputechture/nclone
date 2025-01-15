#include "sim_config.hpp"

SimConfig::SimConfig(bool basicSim,
                     bool fullExport,
                     float tolerance,
                     bool enableAnim,
                     bool logData)
    : basicSim(basicSim),
      fullExport(fullExport),
      tolerance(tolerance),
      enableAnim(enableAnim),
      logData(logData)
{
}

SimConfig SimConfig::fromArgs(const SimConfigArgs *args)
{
  if (!args)
  {
    return SimConfig();
  }

  return SimConfig(
      args->basicSim,
      args->fullExport,
      args->tolerance,
      args->enableAnim,
      args->logData);
}