#include "sim_wrapper.hpp"
#include "ninja.hpp"
#include "entities/entity.hpp"
#include <SFML/Graphics.hpp>

SimWrapper::SimWrapper(bool enableDebugOverlay, bool basicSim, bool fullExport, float tolerance, bool enableAnim, bool logData, const std::string &renderMode)
    : simConfig(basicSim, fullExport, tolerance, enableAnim, logData), renderMode(renderMode)
{
  sim = std::make_unique<Simulation>(simConfig);
  renderer = std::make_unique<Renderer>(sim.get(), enableDebugOverlay, renderMode);
}

void SimWrapper::loadMap(const std::vector<uint8_t> &mapData)
{
  sim->load(mapData);
  renderer->loadTileMap(sim->getTileDic());
}

void SimWrapper::reset()
{
  sim->reset();
}

void SimWrapper::tick(int horInput, int jumpInput)
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

std::vector<float> SimWrapper::getNinjaState() const
{
  auto ninja = sim->getNinja();
  std::vector<float> state;
  state.reserve(10);

  // Match Python implementation order
  state.push_back(ninja->xpos / 1056.0f);                                // Position normalized by screen width
  state.push_back(ninja->ypos / 600.0f);                                 // Position normalized by screen height
  state.push_back((ninja->xspeed / ninja->MAX_HOR_SPEED + 1.0f) / 2.0f); // Speed normalized to [0,1]
  state.push_back((ninja->yspeed / ninja->MAX_HOR_SPEED + 1.0f) / 2.0f);
  state.push_back(ninja->airborn ? 1.0f : 0.0f);
  state.push_back(ninja->walled ? 1.0f : 0.0f);
  state.push_back(static_cast<float>(ninja->jumpDuration) / ninja->MAX_JUMP_DURATION);
  state.push_back((ninja->appliedGravity - ninja->GRAVITY_JUMP) / (ninja->GRAVITY_FALL - ninja->GRAVITY_JUMP));
  state.push_back((ninja->appliedDrag - ninja->DRAG_SLOW) / (ninja->DRAG_REGULAR - ninja->DRAG_SLOW));
  state.push_back((ninja->appliedFriction - ninja->FRICTION_WALL) / (ninja->FRICTION_GROUND - ninja->FRICTION_WALL));

  return state;
}

std::vector<float> SimWrapper::getEntityStates(bool onlyExitAndSwitch) const
{
  std::vector<float> state;

  if (onlyExitAndSwitch)
  {
    // Just return exit switch active and exit door active states
    auto entities = sim->getEntitiesByType(3); // Exit type
    if (!entities.empty())
    {
      auto exitDoor = entities[0];
      state.push_back(exitDoor->isActive() ? 1.0f : 0.0f);
    }

    entities = sim->getEntitiesByType(4); // Switch type
    if (!entities.empty())
    {
      auto exitSwitch = entities[0];
      state.push_back(exitSwitch->isActive() ? 1.0f : 0.0f);
    }
    return state;
  }

  // Constants from Python implementation
  const int MAX_ATTRIBUTES = 4;
  const std::unordered_map<int, int> MAX_COUNTS = {
      {1, 128}, // Toggle Mine
      {2, 128}, // Gold
      {3, 1},   // Exit
      {5, 32},  // Door Regular
      {6, 32},  // Door Locked
      {8, 32},  // Door Trap
      {10, 32}, // Launch Pad
      {11, 32}, // One Way Platform
      {14, 32}, // Drone Zap
      {17, 32}, // Bounce Block
      {20, 32}, // Thwump
      {24, 32}, // Boost Pad
      {25, 32}, // Death Ball
      {26, 32}, // Mini Drone
      {28, 32}  // Shove Thwump
  };

  // Process each entity type
  for (const auto &[entityType, maxCount] : MAX_COUNTS)
  {
    auto entities = sim->getEntitiesByType(entityType);
    state.push_back(static_cast<float>(entities.size()) / maxCount);

    // Process each entity up to maxCount
    for (int i = 0; i < maxCount; i++)
    {
      if (i < entities.size())
      {
        auto entityState = entities[i]->getState(false);
        // Pad with zeros if needed
        while (entityState.size() < MAX_ATTRIBUTES)
        {
          entityState.push_back(0.0f);
        }
        state.insert(state.end(), entityState.begin(), entityState.begin() + MAX_ATTRIBUTES);
      }
      else
      {
        // Add padding for non-existent entity
        state.insert(state.end(), MAX_ATTRIBUTES, 0.0f);
      }
    }
  }

  return state;
}

std::vector<float> SimWrapper::getStateVector(bool onlyExitAndSwitch) const
{
  std::vector<float> state;

  // Add ninja state
  auto ninjaState = getNinjaState();
  state.insert(state.end(), ninjaState.begin(), ninjaState.end());

  // Add entity states
  auto entityStates = getEntityStates(onlyExitAndSwitch);
  state.insert(state.end(), entityStates.begin(), entityStates.end());

  return state;
}

int SimWrapper::getTotalGoldAvailable() const
{
  // Get all entities of type 2 (Gold)
  auto goldEntities = sim->getEntitiesByType(2);
  return goldEntities.size();
}

bool SimWrapper::exitSwitchActivated() const
{
  // Get exit switch (type 4)
  auto entities = sim->getEntitiesByType(4);
  if (!entities.empty())
  {
    auto exitSwitch = entities[0];
    return !exitSwitch->isActive(); // Switch is activated when not active
  }
  return false;
}

std::pair<float, float> SimWrapper::getExitSwitchPosition() const
{
  // Get exit switch (type 4)
  auto entities = sim->getEntitiesByType(4);
  if (!entities.empty())
  {
    auto exitSwitch = entities[0];
    return {exitSwitch->getXPos(), exitSwitch->getYPos()};
  }
  return {0.0f, 0.0f}; // Return origin if no switch exists
}

std::pair<float, float> SimWrapper::getExitDoorPosition() const
{
  // Get exit door (type 3)
  auto entities = sim->getEntitiesByType(3);
  if (!entities.empty())
  {
    auto exitDoor = entities[0];
    return {exitDoor->getXPos(), exitDoor->getYPos()};
  }
  return {0.0f, 0.0f}; // Return origin if no exit door exists
}

bool SimWrapper::isWindowOpen() const
{
  if (renderMode == "human" && renderer)
  {
    return renderer->getWindow().isOpen();
  }
  // If not in human-render mode, or renderer isn't initialized,
  // consider the "window" (conceptually) always open to not break the loop.
  return true;
}

void SimWrapper::render(std::vector<float> &fullBuffer, std::vector<float> &playerViewBuffer,
                        int fullViewWidth, int fullViewHeight,
                        int playerViewWidth, int playerViewHeight)
{
  // Create render texture for off-screen rendering
  sf::RenderTexture renderTexture;
  auto size = renderer->getWindow().getSize();
  if (!renderTexture.resize({size.x, size.y}))
  {
    throw std::runtime_error("Failed to create render texture");
  }

  // Clear with background color
  renderTexture.clear(sf::Color(0xcb, 0xca, 0xd0)); // BG_COLOR

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

  // First, render the full view with downsampling
  {
    // Create a render texture for the downsampled view
    sf::RenderTexture downsampleTexture;
    if (!downsampleTexture.resize({static_cast<unsigned int>(fullViewWidth),
                                   static_cast<unsigned int>(fullViewHeight)}))
    {
      throw std::runtime_error("Failed to create downsample texture");
    }

    // Calculate scale factors
    float scaleX = static_cast<float>(fullViewWidth) / size.x;
    float scaleY = static_cast<float>(fullViewHeight) / size.y;

    // Create sprite with original texture and set scale for Lanczos downsampling
    sf::Sprite windowSprite(windowTexture);
    windowSprite.setScale(sf::Vector2f(scaleX, scaleY));

    // Draw to downsample texture
    downsampleTexture.clear(sf::Color(0xcb, 0xca, 0xd0));
    downsampleTexture.draw(windowSprite);
    downsampleTexture.display();

    // Get the downsampled image
    sf::Image downsampledImage = downsampleTexture.getTexture().copyToImage();

    // Convert to float array [0,1]
    fullBuffer.resize(fullViewWidth * fullViewHeight * 3);
    const uint8_t *pixels = downsampledImage.getPixelsPtr();
    for (size_t i = 0; i < fullViewWidth * fullViewHeight * 4; i += 4)
    {
      size_t j = (i / 4) * 3;
      fullBuffer[j] = pixels[i] / 255.0f;         // R
      fullBuffer[j + 1] = pixels[i + 1] / 255.0f; // G
      fullBuffer[j + 2] = pixels[i + 2] / 255.0f; // B
    }
  }

  // Then, render the player view
  {
    // Get ninja position
    auto ninja = sim->getNinja();
    float ninjaX = ninja->getXPos();
    float ninjaY = ninja->getYPos();

    // Create a view for the cropped region
    sf::View croppedView;
    croppedView.setCenter({ninjaX, ninjaY});
    croppedView.setSize({static_cast<float>(playerViewWidth), static_cast<float>(playerViewHeight)});

    // Set the view and draw
    renderTexture.setView(croppedView);
    sf::Sprite windowSprite(windowTexture);
    renderTexture.draw(windowSprite);
    renderTexture.display();

    // Get the final image
    sf::Image finalImage = renderTexture.getTexture().copyToImage();

    // Convert to float array [0,1]
    playerViewBuffer.resize(playerViewWidth * playerViewHeight * 3);
    const uint8_t *pixels = finalImage.getPixelsPtr();
    for (size_t i = 0; i < playerViewWidth * playerViewHeight * 4; i += 4)
    {
      size_t j = (i / 4) * 3;
      playerViewBuffer[j] = pixels[i] / 255.0f;         // R
      playerViewBuffer[j + 1] = pixels[i + 1] / 255.0f; // G
      playerViewBuffer[j + 2] = pixels[i + 2] / 255.0f; // B
    }
  }
}
