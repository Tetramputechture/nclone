#include "ninja_renderer.hpp"
#include <cmath>

// Initialize static color constant
const sf::Color NinjaRenderer::NINJA_COLOR(0x00, 0x00, 0x00);

// Initialize limb connections
const std::array<std::pair<int, int>, 11> NinjaRenderer::LIMBS = {{{0, 12}, {1, 12}, {2, 8}, {3, 9}, {4, 10}, {5, 11}, {6, 7}, {8, 0}, {9, 0}, {10, 1}, {11, 1}}};

NinjaRenderer::NinjaRenderer()
    : m_ninja(nullptr), m_lines(sf::PrimitiveType::Lines)
{
  m_circle.setFillColor(NINJA_COLOR);
}

void NinjaRenderer::initialize(const Ninja *ninja)
{
  m_ninja = ninja;
  if (m_ninja)
  {
    m_circle.setRadius(m_ninja->RADIUS);
    m_circle.setOrigin({m_ninja->RADIUS, m_ninja->RADIUS});
  }
}

void NinjaRenderer::update()
{
  if (!m_ninja)
    return;

  updateVertices();
}

void NinjaRenderer::updateVertices()
{
  if (!m_ninja)
    return;

  float scale = getScale().x;
  float x = m_ninja->getXPos() * scale;
  float y = m_ninja->getYPos() * scale;

  if (m_ninja->ninjaAnimMode)
  {
    // Clear and resize vertex array for lines
    m_lines.clear();

    // Draw ninja with animation
    const auto &bones = m_ninja->bones;
    for (const auto &[start, end] : LIMBS)
    {
      float x1 = x + bones[start].first * m_ninja->RADIUS * 2 * scale;
      float y1 = y + bones[start].second * m_ninja->RADIUS * 2 * scale;
      float x2 = x + bones[end].first * m_ninja->RADIUS * 2 * scale;
      float y2 = y + bones[end].second * m_ninja->RADIUS * 2 * scale;

      m_lines.append({{x1, y1}, NINJA_COLOR});
      m_lines.append({{x2, y2}, NINJA_COLOR});
    }
  }
  else
  {
    // Update circle position for simple mode
    m_circle.setPosition({x, y});
    m_circle.setScale({scale, scale});
  }
}

void NinjaRenderer::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
  if (!m_ninja)
    return;

  // Apply the transform
  states.transform *= getTransform();

  // Draw either the animated lines or simple circle
  if (m_ninja->ninjaAnimMode)
  {
    target.draw(m_lines, states);
  }
  else
  {
    target.draw(m_circle, states);
  }
}