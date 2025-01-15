#include "renderer.hpp"
#include "ninja.hpp"
#include "physics/segment.hpp"
#include <cmath>
#include <filesystem>

// Initialize static color constants
const sf::Color Renderer::BG_COLOR(0xcb, 0xca, 0xd0);
const sf::Color Renderer::TILE_COLOR(0x79, 0x79, 0x88);
const sf::Color Renderer::NINJA_COLOR(0x00, 0x00, 0x00);

// Initialize entity colors
const std::unordered_map<int, sf::Color> Renderer::ENTITY_COLORS = {
    {1, sf::Color(0x9E, 0x21, 0x26)}, // Toggle Mine
    {2, sf::Color(0xDB, 0xE1, 0x49)}, // Gold
    {3, sf::Color(0x83, 0x83, 0x84)}, // Exit
    {4, sf::Color(0x6D, 0x97, 0xC3)},
    {5, sf::Color(0x00, 0x00, 0x00)}, // Door Regular
    {6, sf::Color(0x00, 0x00, 0x00)}, // Door Locked
    {7, sf::Color(0x00, 0x00, 0x00)},
    {8, sf::Color(0x00, 0x00, 0x00)}, // Door Trap
    {9, sf::Color(0x00, 0x00, 0x00)},
    {10, sf::Color(0x86, 0x87, 0x93)},
    {11, sf::Color(0x66, 0x66, 0x66)},
    {12, sf::Color(0x00, 0x00, 0x00)},
    {13, sf::Color(0x00, 0x00, 0x00)},
    {14, sf::Color(0x6E, 0xC9, 0xE0)},
    {15, sf::Color(0x6E, 0xC9, 0xE0)},
    {16, sf::Color(0x00, 0x00, 0x00)},
    {17, sf::Color(0xE3, 0xE3, 0xE5)},
    {18, sf::Color(0x00, 0x00, 0x00)},
    {19, sf::Color(0x00, 0x00, 0x00)},
    {20, sf::Color(0x83, 0x83, 0x84)},
    {21, sf::Color(0xCE, 0x41, 0x46)},
    {22, sf::Color(0x00, 0x00, 0x00)},
    {23, sf::Color(0x00, 0x00, 0x00)},
    {24, sf::Color(0x66, 0x66, 0x66)},
    {25, sf::Color(0x15, 0xA7, 0xBD)},
    {26, sf::Color(0x6E, 0xC9, 0xE0)},
    {27, sf::Color(0x00, 0x00, 0x00)},
    {28, sf::Color(0x6E, 0xC9, 0xE0)}};

// Initialize limb connections
const std::array<std::pair<int, int>, 11> Renderer::LIMBS = {{{0, 12}, {1, 12}, {2, 8}, {3, 9}, {4, 10}, {5, 11}, {6, 7}, {8, 0}, {9, 0}, {10, 1}, {11, 1}}};

Renderer::Renderer(Simulation *sim, bool enableDebugOverlay)
    : sim(sim),
      window(sf::VideoMode({SRC_WIDTH, SRC_HEIGHT}), "N++ SFML"),
      enableDebugOverlay(enableDebugOverlay)
{
  window.setFramerateLimit(60);

  // Load debug font
  if (!debugFont.openFromFile("assets/fonts/arial.ttf"))
  {
    // Try system font paths
    const std::vector<std::string> fontPaths = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttf"};

    for (const auto &path : fontPaths)
    {
      if (std::filesystem::exists(path) && debugFont.openFromFile(path))
      {
        break;
      }
    }
  }
}

void Renderer::draw(bool init, const std::unordered_map<std::string, float> *debugInfo)
{
  updateScreenSize();
  updateTileOffsets();

  window.clear(TILE_COLOR);
  drawEntities(init);
  drawTiles(init);

  if (enableDebugOverlay && debugInfo)
  {
    drawDebugOverlay(debugInfo);
  }

  window.display();
}

void Renderer::drawCollisionMap(bool init)
{
  updateScreenSize();
  updateTileOffsets();

  window.clear(sf::Color::White);
  drawTiles(init, sf::Color::Black);

  window.display();
}

void Renderer::updateScreenSize()
{
  auto size = window.getSize();
  adjust = std::min(static_cast<float>(size.x) / SRC_WIDTH,
                    static_cast<float>(size.y) / SRC_HEIGHT);
  width = SRC_WIDTH * adjust;
  height = SRC_HEIGHT * adjust;
}

void Renderer::updateTileOffsets()
{
  auto size = window.getSize();
  tileXOffset = (size.x - width) / 2.0f;
  tileYOffset = (size.y - height) / 2.0f;
}

void Renderer::drawTiles(bool init, const sf::Color &tileColor)
{
  const float tileSize = 24.0f * adjust;

  // Create shape for drawing tiles
  sf::RectangleShape tileShape(sf::Vector2f(tileSize, tileSize));
  tileShape.setFillColor(tileColor);

  // Draw tiles
  for (int x = 0; x < 44; ++x)
  {
    for (int y = 0; y < 25; ++y)
    {
      auto tile = sim->getMapData(x + y * 42);
      if (tile == 0)
        continue;

      if (tile == 1 || tile > 33)
      {
        // Full tiles
        tileShape.setPosition(sf::Vector2f(tileXOffset + x * tileSize, tileYOffset + y * tileSize));
        window.draw(tileShape);
      }
      else if (tile < 6)
      {
        // Half tiles
        float dx = (tile == 3) ? tileSize / 2 : 0;
        float dy = (tile == 4) ? tileSize / 2 : 0;
        float w = (tile % 2 == 0) ? tileSize : tileSize / 2;
        float h = (tile % 2 == 0) ? tileSize / 2 : tileSize;

        tileShape.setSize(sf::Vector2f(w, h));
        tileShape.setPosition(sf::Vector2f(tileXOffset + x * tileSize + dx, tileYOffset + y * tileSize + dy));
        window.draw(tileShape);
        tileShape.setSize(sf::Vector2f(tileSize, tileSize)); // Reset size
      }
      else
      {
        // Complex shapes
        drawComplexTile(tile, x, y, tileSize);
      }
    }
  }
}

void Renderer::drawComplexTile(int tileType, float x, float y, float tileSize)
{
  sf::ConvexShape shape;
  float xPos = tileXOffset + x * tileSize;
  float yPos = tileYOffset + y * tileSize;

  if (tileType < 10)
  {
    // Triangle shapes
    shape.setPointCount(3);
    float dx1 = 0;
    float dy1 = tileType == 8 ? tileSize : 0;
    float dx2 = tileType == 9 ? 0 : tileSize;
    float dy2 = tileType == 9 ? tileSize : 0;
    float dx3 = tileType == 6 ? 0 : tileSize;
    float dy3 = tileSize;

    shape.setPoint(0, sf::Vector2f(xPos + dx1, yPos + dy1));
    shape.setPoint(1, sf::Vector2f(xPos + dx2, yPos + dy2));
    shape.setPoint(2, sf::Vector2f(xPos + dx3, yPos + dy3));
  }
  else if (tileType < 14)
  {
    // Quarter moons
    shape.setPointCount(32);
    float centerX = xPos + (tileType == 11 || tileType == 12 ? tileSize : 0);
    float centerY = yPos + (tileType == 12 || tileType == 13 ? tileSize : 0);
    float radius = tileSize;
    float startAngle = (tileType - 10) * M_PI / 2;
    float endAngle = startAngle + M_PI / 2;

    for (int i = 0; i < 32; ++i)
    {
      float angle = startAngle + (endAngle - startAngle) * i / 31.0f;
      shape.setPoint(i, sf::Vector2f(
                            centerX + radius * std::cos(angle),
                            centerY + radius * std::sin(angle)));
    }
  }
  else if (tileType < 18)
  {
    // Quarter pipes
    shape.setPointCount(32);
    float centerX = xPos + (tileType == 15 || tileType == 16 ? tileSize : 0);
    float centerY = yPos + (tileType == 16 || tileType == 17 ? tileSize : 0);
    float radius = tileSize;
    float startAngle = M_PI + (tileType - 14) * M_PI / 2;
    float endAngle = startAngle + M_PI / 2;

    for (int i = 0; i < 32; ++i)
    {
      float angle = startAngle + (endAngle - startAngle) * i / 31.0f;
      shape.setPoint(i, sf::Vector2f(
                            centerX + radius * std::cos(angle),
                            centerY + radius * std::sin(angle)));
    }
  }
  else if (tileType < 22)
  {
    // Short mild slopes
    shape.setPointCount(3);
    float dx1 = 0;
    float dy1 = tileType < 20 ? 0 : tileSize;
    float dx2 = tileSize;
    float dy2 = tileType < 20 ? 0 : tileSize;
    float dx3 = tileType % 2 == 0 ? 0 : tileSize;
    float dy3 = tileSize / 2;

    shape.setPoint(0, sf::Vector2f(xPos + dx1, yPos + dy1));
    shape.setPoint(1, sf::Vector2f(xPos + dx2, yPos + dy2));
    shape.setPoint(2, sf::Vector2f(xPos + dx3, yPos + dy3));
  }
  else if (tileType < 26)
  {
    // Raised mild slopes
    shape.setPointCount(4);
    float dx1 = 0;
    float dy1 = tileType < 24 ? 0 : tileSize;
    float dx2 = tileType % 2 == 0 ? 0 : tileSize;
    float dy2 = tileSize / 2;
    float dx3 = tileSize;
    float dy3 = tileType < 24 ? tileSize / 2 : tileSize;
    float dx4 = tileType % 2 == 0 ? tileSize : 0;
    float dy4 = tileSize;

    shape.setPoint(0, sf::Vector2f(xPos + dx1, yPos + dy1));
    shape.setPoint(1, sf::Vector2f(xPos + dx2, yPos + dy2));
    shape.setPoint(2, sf::Vector2f(xPos + dx3, yPos + dy3));
    shape.setPoint(3, sf::Vector2f(xPos + dx4, yPos + dy4));
  }
  else if (tileType < 30)
  {
    // Short steep slopes
    shape.setPointCount(3);
    float dx1 = tileType < 28 ? tileSize / 2 : 0;
    float dy1 = tileType < 28 ? 0 : tileSize;
    float dx2 = tileType < 28 ? tileSize : tileSize / 2;
    float dy2 = tileType < 28 ? tileSize : 0;
    float dx3 = tileType % 2 == 0 ? tileSize : 0;
    float dy3 = tileType % 2 == 0 ? 0 : tileSize;

    shape.setPoint(0, sf::Vector2f(xPos + dx1, yPos + dy1));
    shape.setPoint(1, sf::Vector2f(xPos + dx2, yPos + dy2));
    shape.setPoint(2, sf::Vector2f(xPos + dx3, yPos + dy3));
  }
  else if (tileType < 34)
  {
    // Raised steep slopes
    shape.setPointCount(4);
    float dx1 = tileType < 32 ? tileSize / 2 : 0;
    float dy1 = tileType < 32 ? 0 : tileSize;
    float dx2 = tileType < 32 ? tileSize : tileSize / 2;
    float dy2 = tileType < 32 ? tileSize : 0;
    float dx3 = tileType % 2 == 0 ? tileSize : 0;
    float dy3 = tileType % 2 == 0 ? 0 : tileSize;
    float dx4 = tileType % 2 == 0 ? 0 : tileSize;
    float dy4 = tileType % 2 == 0 ? tileSize : 0;

    shape.setPoint(0, sf::Vector2f(xPos + dx1, yPos + dy1));
    shape.setPoint(1, sf::Vector2f(xPos + dx2, yPos + dy2));
    shape.setPoint(2, sf::Vector2f(xPos + dx3, yPos + dy3));
    shape.setPoint(3, sf::Vector2f(xPos + dx4, yPos + dy4));
  }

  shape.setFillColor(TILE_COLOR);
  window.draw(shape);
}

void Renderer::drawEntities(bool init)
{
  window.clear(BG_COLOR);

  // Draw segments
  sf::RectangleShape segmentShape;
  segmentShape.setFillColor(TILE_COLOR);

  // Draw active segments
  for (int x = 0; x < 44; ++x)
  {
    for (int y = 0; y < 25; ++y)
    {
      for (const auto &segment : sim->getSegmentsAt({x, y}))
      {
        if (!segment->isActive())
          continue;

        float x1 = segment->getX1() * adjust + tileXOffset;
        float y1 = segment->getY1() * adjust + tileYOffset;
        float x2 = segment->getX2() * adjust + tileXOffset;
        float y2 = segment->getY2() * adjust + tileYOffset;

        if (segment->getType() == "linear")
        {
          float length = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
          float angle = std::atan2(y2 - y1, x2 - x1) * 180 / M_PI;

          segmentShape.setSize(sf::Vector2f(length, SEGMENT_WIDTH * adjust));
          segmentShape.setOrigin(0, SEGMENT_WIDTH * adjust / 2);
          segmentShape.setPosition(x1, y1);
          segmentShape.setRotation(angle);
          window.draw(segmentShape);
        }
        else if (segment->getType() == "circular")
        {
          // Draw circular segments using arc approximation
          sf::VertexArray arc(sf::TriangleStrip);
          float radius = segment->getRadius() * adjust;
          float startAngle = segment->getStartAngle();
          float endAngle = segment->getEndAngle();
          int numPoints = 32;

          for (int i = 0; i <= numPoints; ++i)
          {
            float angle = startAngle + (endAngle - startAngle) * i / numPoints;
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);

            arc.append(sf::Vertex(
                sf::Vector2f(x1 + (radius - SEGMENT_WIDTH * adjust / 2) * cosA,
                             y1 + (radius - SEGMENT_WIDTH * adjust / 2) * sinA),
                TILE_COLOR));
            arc.append(sf::Vertex(
                sf::Vector2f(x1 + (radius + SEGMENT_WIDTH * adjust / 2) * cosA,
                             y1 + (radius + SEGMENT_WIDTH * adjust / 2) * sinA),
                TILE_COLOR));
          }
          window.draw(arc);
        }
      }
    }
  }

  // Draw entities
  sf::CircleShape entityShape;
  for (int type = 1; type < 29; ++type)
  {
    auto it = ENTITY_COLORS.find(type);
    if (it == ENTITY_COLORS.end())
      continue;

    entityShape.setFillColor(it->second);
    for (const auto &entity : sim->getEntitiesByType(type))
    {
      if (!entity->isActive())
        continue;

      float x = entity->getXPos() * adjust + tileXOffset;
      float y = entity->getYPos() * adjust + tileYOffset;

      if (entity->hasNormal())
      {
        // Draw oriented entities
        float radius = entity->getRadius() * adjust;
        float angle = std::atan2(entity->getNormalX(), entity->getNormalY()) + M_PI / 2;

        sf::RectangleShape line(sf::Vector2f(radius * 2, PLATFORM_WIDTH * adjust));
        line.setOrigin(radius, PLATFORM_WIDTH * adjust / 2);
        line.setPosition(x, y);
        line.setRotation(angle * 180 / M_PI);
        line.setFillColor(it->second);
        window.draw(line);
      }
      else
      {
        // Draw regular entities
        float radius = entity->getRadius() * adjust;
        entityShape.setRadius(radius);
        entityShape.setOrigin(radius, radius);
        entityShape.setPosition(x, y);
        window.draw(entityShape);
      }
    }
  }

  // Draw ninja
  drawNinja();
}

void Renderer::drawNinja()
{
  auto ninja = sim->getNinja();
  if (!ninja)
    return;

  float x = ninja->getXPos() * adjust + tileXOffset;
  float y = ninja->getYPos() * adjust + tileYOffset;

  if (sim->getConfig().enableAnim)
  {
    // Draw ninja with animation
    sf::VertexArray lines(sf::Lines);
    const auto &bones = ninja->getBones();

    for (const auto &[start, end] : LIMBS)
    {
      float x1 = x + bones[start].first * ninja->getRadius() * 2 * adjust;
      float y1 = y + bones[start].second * ninja->getRadius() * 2 * adjust;
      float x2 = x + bones[end].first * ninja->getRadius() * 2 * adjust;
      float y2 = y + bones[end].second * ninja->getRadius() * 2 * adjust;

      lines.append(sf::Vertex(sf::Vector2f(x1, y1), NINJA_COLOR));
      lines.append(sf::Vertex(sf::Vector2f(x2, y2), NINJA_COLOR));
    }

    window.draw(lines);
  }
  else
  {
    // Draw ninja as circle
    sf::CircleShape ninjaShape(ninja->getRadius() * adjust);
    ninjaShape.setFillColor(NINJA_COLOR);
    ninjaShape.setOrigin(ninja->getRadius() * adjust, ninja->getRadius() * adjust);
    ninjaShape.setPosition(x, y);
    window.draw(ninjaShape);
  }
}

void Renderer::drawDebugOverlay(const std::unordered_map<std::string, float> *debugInfo)
{
  if (!debugInfo || debugFont.getInfo().family.empty())
  {
    return;
  }

  float yPos = 10;
  const float lineHeight = 20;
  const float xPos = window.getSize().x - 250;

  for (const auto &[key, value] : *debugInfo)
  {
    sf::Text text;
    text.setFont(debugFont);
    text.setString(key + ": " + std::to_string(value));
    text.setCharacterSize(16);
    text.setFillColor(sf::Color(255, 255, 255, 191));
    text.setPosition(xPos, yPos);

    window.draw(text);
    yPos += lineHeight;
  }
}