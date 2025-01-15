#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <array>
#include <unordered_map>
#include <string>
#include <cmath>
#include "simulation.hpp"

class Renderer
{
public:
  static constexpr int SRC_WIDTH = 1056;
  static constexpr int SRC_HEIGHT = 600;
  static constexpr float SEGMENT_WIDTH = 1.0f;
  static constexpr float NINJA_WIDTH = 1.25f;
  static constexpr float DOOR_WIDTH = 2.0f;
  static constexpr float PLATFORM_WIDTH = 3.0f;

  // Constructor
  explicit Renderer(Simulation *sim, bool enableDebugOverlay = false);

  // Main drawing methods
  void draw(bool init = false, const std::unordered_map<std::string, float> *debugInfo = nullptr);
  void drawCollisionMap(bool init = false);

  // Accessors
  sf::RenderWindow &getWindow() { return window; }
  const sf::RenderWindow &getWindow() const { return window; }

private:
  // Helper methods
  void updateScreenSize();
  void updateTileOffsets();
  void drawTiles(bool init = false, const sf::Color &tileColor = sf::Color(0x79, 0x79, 0x88));
  void drawEntities(bool init = false);
  void drawDebugOverlay(const std::unordered_map<std::string, float> *debugInfo);
  void drawComplexTile(int tileType, float x, float y, float tileSize);
  void drawNinja();

  // Member variables
  Simulation *sim;
  sf::RenderWindow window;
  float adjust = 1.0f;
  float width = SRC_WIDTH;
  float height = SRC_HEIGHT;
  float tileXOffset = 0.0f;
  float tileYOffset = 0.0f;
  bool enableDebugOverlay;

  // Static color constants
  static const sf::Color BG_COLOR;
  static const sf::Color TILE_COLOR;
  static const sf::Color NINJA_COLOR;
  static const std::unordered_map<int, sf::Color> ENTITY_COLORS;

  // Ninja limb connections for drawing
  static const std::array<std::pair<int, int>, 11> LIMBS;

  // Font for debug overlay
  sf::Font debugFont;
};