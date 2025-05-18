#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <array>
#include <memory>
#include "utils.hpp"

// TileMap class that inherits from sf::Drawable and sf::Transformable
// This allows us to draw it directly and apply transformations to it
class TileMap : public sf::Drawable, public sf::Transformable
{
public:
  // Constructor takes the tile size and grid dimensions
  TileMap(unsigned int tileSize = 24);

  // Initialize the vertex array with the tile data
  void initialize(const TileDictionary &tileDic);

  // Update a specific tile in the vertex array
  void updateTile(int x, int y, int tileType);

private:
  // Required by sf::Drawable
  virtual void draw(sf::RenderTarget &target, sf::RenderStates states) const override;

  // Helper to set up vertices for a tile
  void setTileVertices(int x, int y, int tileType);

  // Helper to calculate vertices for complex tiles
  void setComplexTileVertices(int tileType, int x, int y);

  // Vertex array storing all the tiles
  sf::VertexArray m_vertices;

  // Size of each tile in pixels
  unsigned int m_tileSize;

  // Grid dimensions
  static constexpr int GRID_WIDTH = 44;
  static constexpr int GRID_HEIGHT = 25;
};