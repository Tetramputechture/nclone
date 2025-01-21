#include "sim_wrapper.hpp"
#include "ninja.hpp"
#include <SFML/Graphics.hpp>

SimWrapper::SimWrapper(bool enableDebugOverlay, bool basicSim, bool fullExport, float tolerance, bool enableAnim, bool logData)
    : simConfig(basicSim, fullExport, tolerance, enableAnim, logData)
{
  sim = std::make_unique<Simulation>(simConfig);
  renderer = std::make_unique<Renderer>(sim.get(), enableDebugOverlay);
}

void SimWrapper::loadMap(const std::vector<uint8_t> &mapData)
{
  sim->load(mapData);
}

void SimWrapper::reset()
{
  sim->reset();
}

void SimWrapper::tick(float horInput, int jumpInput)
{
  sim->tick(horInput, jumpInput);
}

bool SimWrapper::hasWon() const
{
  return sim->getNinja()->hasWon();
}

bool SimWrapper::hasDied() const
{
  return sim->getNinja()->hasDied();
}

std::pair<float, float> SimWrapper::getNinjaPosition() const
{
  auto ninja = sim->getNinja();
  return {ninja->xpos, ninja->ypos};
}

std::pair<float, float> SimWrapper::getNinjaVelocity() const
{
  auto ninja = sim->getNinja();
  return {ninja->xspeed, ninja->yspeed};
}

bool SimWrapper::isNinjaInAir() const
{
  return sim->getNinja()->airborn;
}

bool SimWrapper::isNinjaWalled() const
{
  return sim->getNinja()->walled;
}

int SimWrapper::getGoldCollected() const
{
  return sim->getNinja()->goldCollected;
}

int SimWrapper::getDoorsOpened() const
{
  return sim->getNinja()->doorsOpened;
}

int SimWrapper::getSimFrame() const
{
  return sim->getFrame();
}

void SimWrapper::renderToBuffer(std::vector<float> &buffer)
{
  // Create render texture for off-screen rendering
  sf::RenderTexture renderTexture;
  auto size = renderer->getWindow().getSize();
  if (!renderTexture.resize({size.x, size.y}))
  {
    throw std::runtime_error("Failed to create render texture");
  }

  // Clear with background color
  renderTexture.clear(sf::Color(0xcb, 0xca, 0xd0)); // BG_COLOR from renderer

  // Draw the current frame to the texture
  auto &window = renderer->getWindow();
  renderer->draw(sim->getFrame() <= 1);

  // Capture window contents
  sf::Texture windowTexture;
  if (!windowTexture.resize({window.getSize().x, window.getSize().y}))
  {
    throw std::runtime_error("Failed to create window texture");
  }
  windowTexture.update(window);

  // Copy window contents to render texture
  sf::Sprite windowSprite(windowTexture);
  renderTexture.draw(windowSprite);
  renderTexture.display();

  // Get image from texture
  sf::Image image = renderTexture.getTexture().copyToImage();

  // Convert to float array [0,1]
  auto imageSize = image.getSize();
  buffer.resize(imageSize.x * imageSize.y * 3);

  const uint8_t *pixels = image.getPixelsPtr();
  for (size_t i = 0; i < imageSize.x * imageSize.y * 4; i += 4)
  {
    size_t j = (i / 4) * 3;
    buffer[j] = pixels[i] / 255.0f;         // R
    buffer[j + 1] = pixels[i + 1] / 255.0f; // G
    buffer[j + 2] = pixels[i + 2] / 255.0f; // B
  }
}