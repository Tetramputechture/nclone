#include "door_base.hpp"
#include "../simulation.hpp"
#include "../utils.hpp"
#include "../physics/grid_segment_linear.hpp"
#include <cmath>

DoorBase::DoorBase(int entityType, Simulation *sim, float xcoord, float ycoord,
                   int orientation, float swXcoord, float swYcoord)
    : Entity(entityType, sim, xcoord, ycoord),
      orientation(orientation),
      swXcoord(swXcoord * 6), // Convert to internal coordinates
      swYcoord(swYcoord * 6),
      closed(true)
{
  isVertical = (orientation == 0 || orientation == 4);

  // Get vector from orientation
  auto vec = mapOrientationToVector(orientation);

  // Find the cell that the door is in for the grid segment
  int doorXcell = std::floor((xpos - 12 * vec.first) / 24);
  int doorYcell = std::floor((ypos - 12 * vec.second) / 24);
  auto doorCell = clampCell(doorXcell, doorYcell);

  // Find the half cell of the door for the grid edges
  int doorHalfXcell = 2 * (doorCell.first + 1);
  int doorHalfYcell = 2 * (doorCell.second + 1);

  // Create the grid segment and grid edges
  if (isVertical)
  {
    segment = std::make_shared<GridSegmentLinear>(
        std::make_pair(xpos, ypos - 12),
        std::make_pair(xpos, ypos + 12),
        false // not oriented
    );
    gridEdges.push_back(std::make_pair(doorHalfXcell, doorHalfYcell - 2));
    gridEdges.push_back(std::make_pair(doorHalfXcell, doorHalfYcell - 1));
    for (const auto &edge : gridEdges)
    {
      sim->incrementVerGridEdge(edge, 1);
    }
  }
  else
  {
    segment = std::make_shared<GridSegmentLinear>(
        std::make_pair(xpos - 12, ypos),
        std::make_pair(xpos + 12, ypos),
        false // not oriented
    );
    gridEdges.push_back(std::make_pair(doorHalfXcell - 2, doorHalfYcell));
    gridEdges.push_back(std::make_pair(doorHalfXcell - 1, doorHalfYcell));
    for (const auto &edge : gridEdges)
    {
      sim->incrementHorGridEdge(edge, 1);
    }
  }

  sim->getSegmentsAt(doorCell).push_back(segment);

  // Update position and cell so it corresponds to the switch and not the door
  xpos = swXcoord;
  ypos = swYcoord;
  cell = clampCell(std::floor(xpos / 24), std::floor(ypos / 24));
}

void DoorBase::changeState(bool newClosed)
{
  if (closed != newClosed)
  {
    closed = newClosed;
    segment->setActive(closed);
    logCollision(closed ? 0 : 1);

    for (const auto &edge : gridEdges)
    {
      if (isVertical)
      {
        sim->incrementVerGridEdge(edge, closed ? 1 : -1);
      }
      else
      {
        sim->incrementHorGridEdge(edge, closed ? 1 : -1);
      }
    }
  }
}

std::vector<float> DoorBase::getState(bool minimalState) const
{
  auto baseState = Entity::getState(minimalState);
  if (!minimalState)
  {
    baseState.push_back(static_cast<float>(orientation));
    baseState.push_back(swXcoord);
    baseState.push_back(swYcoord);
    baseState.push_back(closed ? 1.0f : 0.0f);
  }
  return baseState;
}