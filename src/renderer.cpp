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
#include <cstring>

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

// Initialize limb connections
const std::array<std::pair<int, int>, 11> Renderer::LIMBS = {{{0, 12}, {1, 12}, {2, 8}, {3, 9}, {4, 10}, {5, 11}, {6, 7}, {8, 0}, {9, 0}, {10, 1}, {11, 1}}};

// Add at the top with other constants
const int GRID_WIDTH = 44;
const int GRID_HEIGHT = 25;
const float SEGMENT_WIDTH = 1.0f;
const float NINJA_WIDTH = 1.25f;
const float DOOR_WIDTH = 2.0f;
const float PLATFORM_WIDTH = 3.0f;

Renderer::Renderer(Simulation *sim, bool enableDebugOverlay, std::string renderMode)
    : sim(sim),
      window(sf::VideoMode(SRC_WIDTH, SRC_HEIGHT), "NClone Simulation", sf::Style::Default),
      enableDebugOverlay(enableDebugOverlay),
      renderMode(renderMode),
      tileMap(24), // Initialize TileMap with 24x24 tile size
      entityRenderer(sim),
      ninjaRenderer(sim, NINJA_WIDTH)
{
  // If our render mode is human, set the frame limit to 60 and center the window
  if (renderMode == "human")
  {
    // window.setFramerateLimit(60);

    // Center the window on the screen
    sf::Vector2u desktopSize = sf::VideoMode::getDesktopMode().size;
    sf::Vector2i windowPos(
        (desktopSize.x - window.getSize().x) / 2,
        (desktopSize.y - window.getSize().y) / 2);
    window.setPosition(windowPos);
  }

  // Load debug font
  bool fontLoaded = debugFont.loadFromFile("assets/fonts/arial.ttf");
  if (!fontLoaded)
  {
    // Try system font paths
    const std::vector<std::string> fontPaths = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttf",
        "C:/Windows/Fonts/arial.ttf" // Added Windows path
    };

    for (const auto &path : fontPaths)
    {
      if (std::filesystem::exists(path) && debugFont.loadFromFile(path))
      {
        fontLoaded = true;
        break;
      }
    }
  }

  if (fontLoaded)
  {
    debugText.setFont(debugFont);
    debugText.setCharacterSize(16);                        // Default size
    debugText.setFillColor(sf::Color(255, 255, 255, 191)); // Default color
  }
  else
  {
    printf("Error: Could not load any font for debug overlay.\\n");
    // Potentially disable debug overlay features if font is critical
  }

  if (renderMode == "human")
  {
    // Initialize exploration grid texture
    // Ensure window is created and has a size before this
    if (window.getSize().x > 0 && window.getSize().y > 0)
    {
      if (!explorationGridTexture.create(window.getSize().x, window.getSize().y))
      {
        // Handle error: failed to create render texture
        printf("Error: Failed to create explorationGridTexture in constructor\\n");
      }
      else
      {
        explorationGridSprite.setTexture(explorationGridTexture.getTexture());
      }
    }
    else
    {
      printf("Warning: Window size is 0 at constructor, cannot create explorationGridTexture yet.\\n");
    }
  }

  entityRenderer.initialize(sim);
  ninjaRenderer.initialize(sim->getNinja());
}

void Renderer::loadTileMap(const TileDictionary &tileDic)
{
  tileMap.initialize(tileDic);
}

void Renderer::draw(bool init, const std::unordered_map<std::string, float> *debugInfo)
{
  if (window.isOpen())
  {
    sf::Event event;
    while (window.pollEvent(event)) // Changed to while loop for robust event handling
    {
      if (event.type == sf::Event::Closed)
      {
        window.close();
      }
      else if (event.type == sf::Event::KeyPressed)
      {
        if (event.key.scancode == sf::Keyboard::Scancode::Escape) //
          window.close();
      }
      // Handle other events like Resized if necessary
      if (event.type == sf::Event::Resized)
      {
        // Update view to match new window size
        sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
        window.setView(sf::View(visibleArea));
        // updateScreenSize will be called next, which should handle adjustments
      }
    }
    updateScreenSize();  // This should use window.getSize()
    updateTileOffsets(); // This also depends on window.getSize()

    // Apply screen scaling to tilemap, entity renderer and ninja renderer
    tileMap.setScale(sf::Vector2f(adjust, adjust));
    tileMap.setPosition(sf::Vector2f(tileXOffset, tileYOffset));

    entityRenderer.setScale(sf::Vector2f(adjust, adjust));
    entityRenderer.setPosition(sf::Vector2f(tileXOffset, tileYOffset));

    ninjaRenderer.setScale(sf::Vector2f(adjust, adjust));
    ninjaRenderer.setPosition(sf::Vector2f(tileXOffset, tileYOffset));

    // Ensure the exploration grid texture is created/resized with the current window size
    sf::Vector2u currentWindowSize = window.getSize();
    if (currentWindowSize.x > 0 && currentWindowSize.y > 0)
    {
      if (explorationGridTexture.getSize().x != currentWindowSize.x || explorationGridTexture.getSize().y != currentWindowSize.y)
      {
        if (!explorationGridTexture.create(currentWindowSize.x, currentWindowSize.y))
        {
          printf("Error: Failed to create/resize explorationGridTexture in draw()\\n");
          // Decide how to handle this: return, throw, or attempt to continue without this texture
        }
        else
        {
          explorationGridSprite.setTexture(explorationGridTexture.getTexture(), true); // Reset texture rect
        }
      }
    }

    // Draw background
    window.clear(BG_COLOR);

    // Update and draw entities
    entityRenderer.update();
    window.draw(entityRenderer);

    // Draw tilemap
    window.draw(tileMap);

    // Update and draw ninja
    ninjaRenderer.update();
    window.draw(ninjaRenderer);

    if (enableDebugOverlay && debugInfo)
    {
      drawExplorationGrid(debugInfo);
      drawDebugOverlay(debugInfo);
    }

    if (renderMode == "human")
    {
      window.display();
    }
  }
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

void Renderer::drawExplorationGrid(const std::unordered_map<std::string, float> *debugInfo)
{
  if (!debugInfo)
  {
    return;
  }

  // Ensure texture is the correct size
  sf::Vector2u currentWindowSize = window.getSize();
  if (currentWindowSize.x == 0 || currentWindowSize.y == 0)
  {
    printf("Error: Window size is zero in drawExplorationGrid.\\n");
    return;
  }

  if (explorationGridTexture.getSize().x != currentWindowSize.x || explorationGridTexture.getSize().y != currentWindowSize.y)
  {
    if (!explorationGridTexture.create(currentWindowSize.x, currentWindowSize.y))
    {
      printf("Error: Failed to create/resize explorationGridTexture in drawExplorationGrid()\\n");
      return;
    }
    explorationGridSprite.setTexture(explorationGridTexture.getTexture(), true);
  }

  explorationGridTexture.clear(sf::Color::Transparent);

  // Calculate cell size
  float cellSize = 24.0f * adjust;
  float quarterSize = cellSize / 2.0f;

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
      explorationGridTexture.draw(cell);
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
    explorationGridTexture.draw(line);
  }

  for (int i = 0; i <= GRID_HEIGHT; i++)
  {
    float y = i * cellSize;
    sf::RectangleShape line;
    line.setSize({GRID_WIDTH * cellSize, 2.0f});
    line.setPosition({0, y});
    line.setFillColor(sf::Color(255, 255, 255, 64));
    explorationGridTexture.draw(line);
  }

  explorationGridTexture.display();
  explorationGridSprite.setPosition(0.f, 0.f); // Corrected to use float arguments
  window.draw(explorationGridSprite);
}

void Renderer::drawDebugOverlay(const std::unordered_map<std::string, float> *debugInfo)
{
  if (!debugInfo || !debugFont.getInfo().family.empty()) // Corrected logic: ensure font IS loaded
  {
    // Check if font is actually loaded before trying to use it.
    if (debugFont.getInfo().family.empty())
    {
      // Optional: print a warning or skip drawing if font not loaded.
      // printf("Warning: Debug font not loaded, cannot draw debug overlay.\\n");
      return;
    }
  }
  else
  { // If debugInfo is null or font is not loaded
    return;
  }

  float yPos = 10;
  const float lineHeight = 20;
  const float xPos = window.getSize().x - 250;

  for (const auto &[key, value] : *debugInfo)
  {
    debugText.setString(key + ": " + std::to_string(value));
    debugText.setPosition({xPos, yPos});
    window.draw(debugText);
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