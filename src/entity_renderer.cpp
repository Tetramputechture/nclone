#include "entity_renderer.hpp"
#include "entities/gold.hpp"
#include "entities/exit_door.hpp"
#include "entities/door_regular.hpp"
#include "entities/drone_base.hpp"
#include "entities/laser.hpp"
#include <cmath>

// Initialize static color constants - copied from renderer.cpp
const std::unordered_map<int, sf::Color> EntityRenderer::ENTITY_COLORS = {
    {1, sf::Color(0x9E, 0x21, 0x26)}, // Toggle Mine
    {2, sf::Color(0xDB, 0xE1, 0x49)}, // Gold
    {3, sf::Color(0x83, 0x83, 0x84)}, // Exit
    {4, sf::Color(0x6D, 0x97, 0xC3)},
    {5, sf::Color(0x00, 0x00, 0x00)}, // Door Regular
    {6, sf::Color(0x00, 0x00, 0x00)}, // Door Locked
    {7, sf::Color(0x00, 0x00, 0x00)},
    {8, sf::Color(0x00, 0x00, 0x00)}, // Door Trap
    {9, sf::Color(0x00, 0x00, 0x00)},
    {10, sf::Color(0x86, 0x87, 0x93)}, // Launch Pad
    {11, sf::Color(0x66, 0x66, 0x66)}, // One Way Platform
    {12, sf::Color(0x00, 0x00, 0x00)},
    {13, sf::Color(0x00, 0x00, 0x00)},
    {14, sf::Color(0x6E, 0xC9, 0xE0)}, // Drone Zap
    {15, sf::Color(0x6E, 0xC9, 0xE0)}, // Drone Chaser
    {16, sf::Color(0x00, 0x00, 0x00)},
    {17, sf::Color(0xE3, 0xE3, 0xE5)}, // Bounce Block
    {18, sf::Color(0x00, 0x00, 0x00)},
    {19, sf::Color(0x00, 0x00, 0x00)},
    {20, sf::Color(0x83, 0x83, 0x84)}, // Thwump
    {21, sf::Color(0xCE, 0x41, 0x46)}, // Toggle Mine (toggled state)
    {22, sf::Color(0x00, 0x00, 0x00)},
    {23, sf::Color(0x00, 0x00, 0x00)}, // Laser
    {24, sf::Color(0x66, 0x66, 0x66)}, // Boost Pad
    {25, sf::Color(0x15, 0xA7, 0xBD)}, // Death Ball
    {26, sf::Color(0x6E, 0xC9, 0xE0)}, // Mini Drone
    {27, sf::Color(0x00, 0x00, 0x00)},
    {28, sf::Color(0x6E, 0xC9, 0xE0)} // Shove Thwump
};

EntityRenderer::EntityRenderer() : m_sim(nullptr)
{
}

void EntityRenderer::initialize(const Simulation *sim)
{
  m_sim = sim;
}

void EntityRenderer::update()
{
  // Clear existing circles
  m_circles.clear();

  // Get the current scale factor from the transform
  float adjust = getScale().x;

  // Process each entity type
  for (int type = 1; type < 29; ++type)
  {
    auto it = ENTITY_COLORS.find(type);
    if (it == ENTITY_COLORS.end())
      continue;

    for (const auto &entity : m_sim->getEntitiesByType(type))
    {
      if (!entity->isActive())
        continue;

      addEntityToCircles(entity.get(), adjust);
    }
  }
}

void EntityRenderer::addEntityToCircles(const Entity *entity, float adjust)
{
  float x = entity->getXPos() * adjust;
  float y = entity->getYPos() * adjust;

  // Determine radius based on entity type
  float radius = DEFAULT_RADIUS * adjust;
  if (auto gold = dynamic_cast<const Gold *>(entity))
    radius = gold->RADIUS * adjust;
  else if (auto exit = dynamic_cast<const ExitDoor *>(entity))
    radius = exit->RADIUS * adjust;
  else if (auto door = dynamic_cast<const DoorRegular *>(entity))
    radius = door->RADIUS * adjust;
  else if (auto drone = dynamic_cast<const DroneBase *>(entity))
    radius = drone->RADIUS * adjust;
  else if (auto laser = dynamic_cast<const Laser *>(entity))
    radius = laser->RADIUS * adjust;

  // Get entity color
  sf::Color color = ENTITY_COLORS.at(entity->getEntityType());

  // Create and configure circle shape
  sf::CircleShape circle(radius);
  circle.setFillColor(color);
  circle.setPointCount(15);
  circle.setPosition({x - radius, y - radius}); // Adjust position to center the circle

  // Add circle to vector
  m_circles.push_back(std::move(circle));
}

void EntityRenderer::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
  // Apply the transform
  states.transform *= getTransform();

  // Draw all circles
  for (const auto &circle : m_circles)
  {
    target.draw(circle, states);
  }
}