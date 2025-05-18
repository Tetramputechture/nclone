#pragma once

#include <SFML/Graphics.hpp>
#include "simulation.hpp"
#include <unordered_map>
#include <vector>

class EntityRenderer : public sf::Drawable, public sf::Transformable
{
public:
  explicit EntityRenderer();

  // Initialize/update the circle shapes with entity data
  void initialize(const Simulation *sim);
  void update();

private:
  virtual void draw(sf::RenderTarget &target, sf::RenderStates states) const override;

  // Vector to store all entity circle shapes
  std::vector<sf::CircleShape> m_circles;
  // std::vector<sf::RectangleShape> m_rectangles; // For rectangular entities like doors
  // std::vector<sf::VertexArray> m_lines;      // For line-based entities like lasers or oriented platforms

  // Store the simulation pointer
  const Simulation *m_sim;

  // Helper methods
  void addEntityToCircles(const Entity *entity, float adjust); // Reverted name

  // Constants for entity rendering
  static constexpr float DEFAULT_RADIUS = 10.0f;
  static const std::unordered_map<int, sf::Color> ENTITY_COLORS;
};