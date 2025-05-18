#pragma once

#include <SFML/Graphics.hpp>
#include "ninja.hpp"

class NinjaRenderer : public sf::Drawable, public sf::Transformable
{
public:
  explicit NinjaRenderer();

  // Initialize/update the vertices with ninja data
  void initialize(const Ninja *ninja);
  void update();

private:
  virtual void draw(sf::RenderTarget &target, sf::RenderStates states) const override;

  // Store the ninja pointer
  const Ninja *m_ninja;

  // Store vertices for drawing
  sf::VertexArray m_lines;
  sf::CircleShape m_circle;

  // Helper methods
  void updateVertices();

  // Constants for ninja rendering
  static const sf::Color NINJA_COLOR;
  static const std::array<std::pair<int, int>, 11> LIMBS;
};