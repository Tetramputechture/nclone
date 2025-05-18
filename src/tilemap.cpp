#include "tilemap.hpp"
#include <cmath>

TileMap::TileMap(unsigned int tileSize)
    : m_tileSize(tileSize)
{
  // Initialize vertex array to hold all tiles
  // We use triangles as the primitive type for maximum flexibility with complex tiles
  m_vertices.setPrimitiveType(sf::PrimitiveType::Triangles);

  // Each tile can potentially use up to 8 triangles (24 vertices) for complex shapes
  // This pre-allocates enough space for the worst case
  m_vertices.resize(GRID_WIDTH * GRID_HEIGHT * 24);
}

void TileMap::initialize(const TileDictionary &tileDic)
{
  // Clear existing vertices
  m_vertices.clear();

  // Process each tile in the grid
  for (int y = 0; y < GRID_HEIGHT; ++y)
  {
    for (int x = 0; x < GRID_WIDTH; ++x)
    {
      auto it = tileDic.find({x, y});
      if (it != tileDic.end())
      {
        setTileVertices(x, y, it->second);
      }
    }
  }
}

void TileMap::setTileVertices(int x, int y, int tileType)
{
  // Base coordinates for the tile
  float xPos = x * m_tileSize;
  float yPos = y * m_tileSize;

  // Color for the tile (using the existing color scheme)
  sf::Color tileColor(0x79, 0x79, 0x88);

  if (tileType == 1)
  {
    // Full tile - two triangles
    sf::Vertex triangle1[] = {
        {sf::Vector2f(xPos, yPos), tileColor},
        {sf::Vector2f(xPos + m_tileSize, yPos), tileColor},
        {sf::Vector2f(xPos, yPos + m_tileSize), tileColor}};

    sf::Vertex triangle2[] = {
        {sf::Vector2f(xPos + m_tileSize, yPos), tileColor},
        {sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize), tileColor},
        {sf::Vector2f(xPos, yPos + m_tileSize), tileColor}};

    // Add vertices to the array
    for (int i = 0; i < 3; ++i)
    {
      m_vertices.append(triangle1[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
      m_vertices.append(triangle2[i]);
    }
  }
  else if (tileType >= 2 && tileType <= 5)
  {
    // Half tiles
    float dx = (tileType == 3) ? m_tileSize / 2 : 0;
    float dy = (tileType == 4) ? m_tileSize / 2 : 0;
    float w = (tileType % 2 == 0) ? m_tileSize : m_tileSize / 2;
    float h = (tileType % 2 == 0) ? m_tileSize / 2 : m_tileSize;

    sf::Vertex triangle1[] = {
        {sf::Vector2f(xPos + dx, yPos + dy), tileColor},
        {sf::Vector2f(xPos + dx + w, yPos + dy), tileColor},
        {sf::Vector2f(xPos + dx, yPos + dy + h), tileColor}};

    sf::Vertex triangle2[] = {
        {sf::Vector2f(xPos + dx + w, yPos + dy), tileColor},
        {sf::Vector2f(xPos + dx + w, yPos + dy + h), tileColor},
        {sf::Vector2f(xPos + dx, yPos + dy + h), tileColor}};

    for (int i = 0; i < 3; ++i)
    {
      m_vertices.append(triangle1[i]);
      m_vertices.append(triangle2[i]);
    }
  }
  else if (tileType > 5)
  {
    // Complex tiles (slopes, curves, etc.)
    setComplexTileVertices(tileType, x, y);
  }
}

void TileMap::setComplexTileVertices(int tileType, int x, int y)
{
  float xPos = x * m_tileSize;
  float yPos = y * m_tileSize;
  sf::Color tileColor(0x79, 0x79, 0x88);

  if (tileType >= 6 && tileType <= 9)
  {
    // 45-degree slopes
    std::array<sf::Vector2f, 3> points;

    switch (tileType)
    {
    case 6: // Bottom-left to top-right
      points = {
          sf::Vector2f(xPos, yPos + m_tileSize),
          sf::Vector2f(xPos + m_tileSize, yPos),
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize)};
      break;
    case 7: // Top-left to bottom-right
      points = {
          sf::Vector2f(xPos, yPos),
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize),
          sf::Vector2f(xPos, yPos + m_tileSize)};
      break;
    case 8: // Top-right to bottom-left
      points = {
          sf::Vector2f(xPos + m_tileSize, yPos),
          sf::Vector2f(xPos, yPos + m_tileSize),
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize)};
      break;
    case 9: // Bottom-right to top-left
      points = {
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize),
          sf::Vector2f(xPos, yPos),
          sf::Vector2f(xPos, yPos + m_tileSize)};
      break;
    }

    for (const auto &point : points)
    {
      m_vertices.append({point, tileColor});
    }
  }
  else if (tileType >= 10 && tileType <= 13)
  {
    // Quarter moons - approximate with triangle fan
    const int numSegments = 16;
    float centerX, centerY;
    float startAngle;

    switch (tileType)
    {
    case 10: // Bottom-left
      centerX = xPos;
      centerY = yPos;
      startAngle = 0;
      break;
    case 11: // Bottom-right
      centerX = xPos + m_tileSize;
      centerY = yPos;
      startAngle = M_PI / 2;
      break;
    case 12: // Top-right
      centerX = xPos + m_tileSize;
      centerY = yPos + m_tileSize;
      startAngle = M_PI;
      break;
    case 13: // Top-left
      centerX = xPos;
      centerY = yPos + m_tileSize;
      startAngle = 3 * M_PI / 2;
      break;
    }

    // Create triangle fan
    sf::Vector2f center(centerX, centerY);
    float angleStep = M_PI / 2 / numSegments;

    for (int i = 0; i < numSegments; i++)
    {
      float angle1 = startAngle + i * angleStep;
      float angle2 = startAngle + (i + 1) * angleStep;

      m_vertices.append({center, tileColor});
      m_vertices.append({sf::Vector2f(centerX + m_tileSize * std::cos(angle1),
                                      centerY + m_tileSize * std::sin(angle1)),
                         tileColor});
      m_vertices.append({sf::Vector2f(centerX + m_tileSize * std::cos(angle2),
                                      centerY + m_tileSize * std::sin(angle2)),
                         tileColor});
    }
  }
  else if (tileType >= 14 && tileType <= 17)
  {
    // Quarter pipes - similar to quarter moons but concave
    const int numSegments = 16;
    float centerX, centerY;
    float startAngle;

    switch (tileType)
    {
    case 14: // Top-right
      centerX = xPos + m_tileSize;
      centerY = yPos + m_tileSize;
      startAngle = M_PI;
      break;
    case 15: // Top-left
      centerX = xPos;
      centerY = yPos + m_tileSize;
      startAngle = 3 * M_PI / 2;
      break;
    case 16: // Bottom-left
      centerX = xPos;
      centerY = yPos;
      startAngle = 0;
      break;
    case 17: // Bottom-right
      centerX = xPos + m_tileSize;
      centerY = yPos;
      startAngle = M_PI / 2;
      break;
    }

    // Create concave quarter circle using triangle strip
    float angleStep = M_PI / 2 / numSegments;

    for (int i = 0; i <= numSegments; i++)
    {
      float angle = startAngle + i * angleStep;
      float cos = std::cos(angle);
      float sin = std::sin(angle);

      m_vertices.append({sf::Vector2f(centerX + m_tileSize * 0.7f * cos,
                                      centerY + m_tileSize * 0.7f * sin),
                         tileColor});
      m_vertices.append({sf::Vector2f(centerX + m_tileSize * cos,
                                      centerY + m_tileSize * sin),
                         tileColor});
    }
  }
  else if ((tileType >= 18 && tileType <= 21) || // Short mild slopes
           (tileType >= 22 && tileType <= 25) || // Raised mild slopes
           (tileType >= 26 && tileType <= 29) || // Short steep slopes
           (tileType >= 30 && tileType <= 33))   // Raised steep slopes
  {
    // All slope types use triangles, just with different vertex positions
    std::array<sf::Vector2f, 3> points;
    float midX, midY;

    if (tileType >= 18 && tileType <= 21) // Short mild slopes
    {
      midX = m_tileSize / 2;
      midY = m_tileSize / 2;
    }
    else if (tileType >= 22 && tileType <= 25) // Raised mild slopes
    {
      midX = m_tileSize / 2;
      midY = m_tileSize * 0.75f;
    }
    else if (tileType >= 26 && tileType <= 29) // Short steep slopes
    {
      midX = m_tileSize / 3;
      midY = m_tileSize / 3;
    }
    else // Raised steep slopes
    {
      midX = m_tileSize / 3;
      midY = m_tileSize * 0.75f;
    }

    switch (tileType % 4)
    {
    case 2: // Bottom-left to top-right
      points = {
          sf::Vector2f(xPos, yPos + m_tileSize),
          sf::Vector2f(xPos + m_tileSize, yPos),
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize)};
      break;
    case 3: // Top-left to bottom-right
      points = {
          sf::Vector2f(xPos, yPos),
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize),
          sf::Vector2f(xPos, yPos + m_tileSize)};
      break;
    case 0: // Top-right to bottom-left
      points = {
          sf::Vector2f(xPos + m_tileSize, yPos),
          sf::Vector2f(xPos, yPos + m_tileSize),
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize)};
      break;
    case 1: // Bottom-right to top-left
      points = {
          sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize),
          sf::Vector2f(xPos, yPos),
          sf::Vector2f(xPos, yPos + m_tileSize)};
      break;
    }

    for (const auto &point : points)
    {
      m_vertices.append({point, tileColor});
    }
  }
  else if (tileType >= 34 && tileType <= 37) // Glitched tiles
  {
    // Glitched tiles are partial tiles with only one edge
    sf::Vertex triangle1[3], triangle2[3];

    switch (tileType)
    {
    case 34: // Top edge
      triangle1[0] = {sf::Vector2f(xPos, yPos), tileColor};
      triangle1[1] = {sf::Vector2f(xPos + m_tileSize, yPos), tileColor};
      triangle1[2] = {sf::Vector2f(xPos + m_tileSize / 2, yPos + m_tileSize / 4), tileColor};
      break;
    case 35: // Bottom edge
      triangle1[0] = {sf::Vector2f(xPos, yPos + m_tileSize), tileColor};
      triangle1[1] = {sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize), tileColor};
      triangle1[2] = {sf::Vector2f(xPos + m_tileSize / 2, yPos + m_tileSize * 3 / 4), tileColor};
      break;
    case 36: // Right edge
      triangle1[0] = {sf::Vector2f(xPos + m_tileSize, yPos), tileColor};
      triangle1[1] = {sf::Vector2f(xPos + m_tileSize, yPos + m_tileSize), tileColor};
      triangle1[2] = {sf::Vector2f(xPos + m_tileSize * 3 / 4, yPos + m_tileSize / 2), tileColor};
      break;
    case 37: // Left edge
      triangle1[0] = {sf::Vector2f(xPos, yPos), tileColor};
      triangle1[1] = {sf::Vector2f(xPos, yPos + m_tileSize), tileColor};
      triangle1[2] = {sf::Vector2f(xPos + m_tileSize / 4, yPos + m_tileSize / 2), tileColor};
      break;
    }

    for (int i = 0; i < 3; ++i)
    {
      m_vertices.append(triangle1[i]);
    }
  }
}

void TileMap::updateTile(int x, int y, int tileType)
{
  // Calculate the base index for this tile in the vertex array
  // This needs to be adjusted based on how many vertices each tile type uses
  setTileVertices(x, y, tileType);
}

void TileMap::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
  // Apply the transform
  states.transform *= getTransform();

  // Draw the vertex array
  target.draw(m_vertices, states);
}