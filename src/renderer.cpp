#include "renderer.hpp"
#include "entities/entity.hpp"
#include "entities/gold.hpp"
#include "entities/exit_door.hpp"
#include "entities/door_base.hpp"
#include "entities/drone_base.hpp"
#include "entities/laser.hpp"
#include "ninja.hpp"
#include "physics/segment.hpp"
#include "entities/door_regular.hpp"
#include <cmath>
#include <filesystem>

// Initialize static color constants
const sf::Color Renderer::BG_COLOR(0xcb, 0xca, 0xd0);
const sf::Color Renderer::TILE_COLOR(0x79, 0x79, 0x88);
const sf::Color Renderer::NINJA_COLOR(0x00, 0x00, 0x00);

// Exploration grid colors
const sf::Color Renderer::CELL_COLOR(0, 0, 0, 0);             // Transparent
const sf::Color Renderer::CELL_VISITED_COLOR(0, 255, 0, 192); // Bright green with 75% opacity
const sf::Color Renderer::GRID_CELL_COLOR(255, 255, 255, 64); // White with 25% opacity

// Area base colors
const sf::Color Renderer::AREA_4X4_COLOR(255, 50, 50);     // Base red for 4x4
const sf::Color Renderer::AREA_8X8_COLOR(50, 50, 255);     // Base blue for 8x8
const sf::Color Renderer::AREA_16X16_COLOR(128, 128, 128); // Base grey for 16x16

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
    {28, sf::Color(0x6E, 0xC9, 0xE0)}}; // Shove Thwump

// Initialize limb connections
const std::array<std::pair<int, int>, 11> Renderer::LIMBS = {{{0, 12}, {1, 12}, {2, 8}, {3, 9}, {4, 10}, {5, 11}, {6, 7}, {8, 0}, {9, 0}, {10, 1}, {11, 1}}};

// Add at the top with other constants
const int GRID_WIDTH = 44;
const int GRID_HEIGHT = 25;
const float SEGMENT_WIDTH = 1.0f;
const float NINJA_WIDTH = 1.25f;
const float DOOR_WIDTH = 2.0f;
const float PLATFORM_WIDTH = 3.0f;

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
    drawExplorationGrid(debugInfo);
    drawDebugOverlay(debugInfo);
  }

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
        tileShape.setPosition({tileXOffset + x * tileSize, tileYOffset + y * tileSize});
        window.draw(tileShape);
      }
      else if (tile < 6)
      {
        // Half tiles
        float dx = (tile == 3) ? tileSize / 2 : 0;
        float dy = (tile == 4) ? tileSize / 2 : 0;
        float w = (tile % 2 == 0) ? tileSize : tileSize / 2;
        float h = (tile % 2 == 0) ? tileSize / 2 : tileSize;

        tileShape.setSize({w, h});
        tileShape.setPosition({tileXOffset + x * tileSize + dx, tileYOffset + y * tileSize + dy});
        window.draw(tileShape);
        tileShape.setSize({tileSize, tileSize}); // Reset size
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

          segmentShape.setSize({length, SEGMENT_WIDTH * adjust});
          segmentShape.setOrigin({0, SEGMENT_WIDTH * adjust / 2});
          segmentShape.setPosition({x1, y1});
          segmentShape.setRotation(sf::degrees(angle));
          window.draw(segmentShape);
        }
        else if (segment->getType() == "circular")
        {
          // Draw circular segments using arc approximation
          sf::VertexArray arc(sf::PrimitiveType::TriangleStrip);
          float radius = segment->getRadius() * adjust;
          float startAngle = segment->getStartAngle();

          for (float angle = startAngle; angle <= startAngle + M_PI; angle += M_PI / 16)
          {
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);

            arc.append({{x1 + (radius - SEGMENT_WIDTH * adjust / 2) * cosA,
                         y1 + (radius - SEGMENT_WIDTH * adjust / 2) * sinA},
                        TILE_COLOR});

            arc.append({{x1 + (radius + SEGMENT_WIDTH * adjust / 2) * cosA,
                         y1 + (radius + SEGMENT_WIDTH * adjust / 2) * sinA},
                        TILE_COLOR});
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

      // Draw regular entities
      float radius = 10.0f * adjust; // Default radius
      if (auto gold = dynamic_cast<const Gold *>(entity.get()))
        radius = gold->RADIUS * adjust;
      else if (auto exit = dynamic_cast<const ExitDoor *>(entity.get()))
        radius = exit->RADIUS * adjust;
      else if (auto door = dynamic_cast<const DoorRegular *>(entity.get()))
        radius = door->RADIUS * adjust;
      else if (auto drone = dynamic_cast<const DroneBase *>(entity.get()))
        radius = drone->RADIUS * adjust;
      else if (auto laser = dynamic_cast<const Laser *>(entity.get()))
        radius = laser->RADIUS * adjust;

      entityShape.setRadius(radius);
      entityShape.setOrigin({radius, radius});
      entityShape.setPosition({x, y});
      window.draw(entityShape);
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
    sf::VertexArray lines(sf::PrimitiveType::Lines);
    const auto &bones = ninja->bones;

    for (const auto &[start, end] : LIMBS)
    {
      float x1 = x + bones[start].first * ninja->RADIUS * 2 * adjust;
      float y1 = y + bones[start].second * ninja->RADIUS * 2 * adjust;
      float x2 = x + bones[end].first * ninja->RADIUS * 2 * adjust;
      float y2 = y + bones[end].second * ninja->RADIUS * 2 * adjust;

      lines.append({{x1, y1}, NINJA_COLOR});
      lines.append({{x2, y2}, NINJA_COLOR});
    }

    window.draw(lines);
  }
  else
  {
    // Draw ninja as circle
    sf::CircleShape ninjaShape(ninja->RADIUS * adjust);
    ninjaShape.setFillColor(NINJA_COLOR);
    ninjaShape.setOrigin({ninja->RADIUS * adjust, ninja->RADIUS * adjust});
    ninjaShape.setPosition({x, y});
    window.draw(ninjaShape);
  }
}

void Renderer::drawExplorationGrid(const std::unordered_map<std::string, float> *debugInfo)
{
  if (!debugInfo)
  {
    return;
  }

  // Calculate cell size
  float cellSize = 24.0f * adjust;
  float quarterSize = cellSize / 2.0f;

  // Create render texture for exploration grid
  sf::RenderTexture gridTexture;
  if (!gridTexture.resize({static_cast<unsigned int>(window.getSize().x),
                           static_cast<unsigned int>(window.getSize().y)}))
  {
    return;
  }
  gridTexture.clear(sf::Color::Transparent);

  // Draw grid cells
  for (int y = 0; y < GRID_HEIGHT; y++)
  {
    for (int x = 0; x < GRID_WIDTH; x++)
    {
      float xPos = x * cellSize;
      float yPos = y * cellSize;

      // Draw cell background
      sf::RectangleShape cell;
      cell.setSize({cellSize, cellSize});
      cell.setPosition({xPos, yPos});
      cell.setFillColor(GRID_CELL_COLOR);
      gridTexture.draw(cell);
    }
  }

  // Draw grid lines
  for (int i = 0; i <= GRID_WIDTH; i++)
  {
    float x = i * cellSize;
    sf::RectangleShape line;
    line.setSize({2.0f, GRID_HEIGHT * cellSize});
    line.setPosition({x, 0});
    line.setFillColor(sf::Color(255, 255, 255, 64));
    gridTexture.draw(line);
  }

  for (int i = 0; i <= GRID_HEIGHT; i++)
  {
    float y = i * cellSize;
    sf::RectangleShape line;
    line.setSize({GRID_WIDTH * cellSize, 2.0f});
    line.setPosition({0, y});
    line.setFillColor(sf::Color(255, 255, 255, 64));
    gridTexture.draw(line);
  }

  gridTexture.display();
  sf::Sprite gridSprite(gridTexture.getTexture());
  window.draw(gridSprite);
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
    sf::Text text(debugFont);
    text.setString(key + ": " + std::to_string(value));
    text.setCharacterSize(16);
    text.setFillColor(sf::Color(255, 255, 255, 191));
    text.setPosition({xPos, yPos});

    window.draw(text);
    yPos += lineHeight;
  }
}

sf::Color Renderer::getAreaColor(const sf::Color &baseColor, int index, int maxIndex, uint8_t opacity) const
{
  // Calculate brightness factor (0.3 to 1.0)
  float brightness = 1.0f - (0.7f * static_cast<float>(index) / maxIndex);
  return sf::Color(
      static_cast<uint8_t>(baseColor.r * brightness),
      static_cast<uint8_t>(baseColor.g * brightness),
      static_cast<uint8_t>(baseColor.b * brightness),
      opacity);
}