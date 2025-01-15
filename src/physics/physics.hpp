#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <array>
#include <tuple>
#include <optional>
#include <algorithm>
#include <string>

// Forward declarations
class Simulation;
class Ninja;

class Physics
{
public:
    // Utility functions
    static float clamp(float n, float a, float b);
    static std::pair<int, int> clampCell(int xcell, int ycell);
    static std::pair<int, int> clampHalfCell(int xcell, int ycell);
    static int16_t packCoord(float coord);

    // Cache management
    static void clearCaches();

    // Geometry calculations
    static std::vector<std::tuple<float, float>> gatherSegmentsFromRegion(
        const Simulation &sim, float x1, float y1, float x2, float y2);

    static std::vector<std::tuple<float, float>> gatherEntitiesFromNeighbourhood(
        const Simulation &sim, float xpos, float ypos);

    static float sweepCircleVsTiles(
        const Simulation &sim, float xposOld, float yposOld,
        float dx, float dy, float radius);

    static std::optional<std::tuple<std::pair<float, float>, std::pair<float, float>>> penetrationSquareVsPoint(float sXpos, float sYpos, float pXpos, float pYpos, float semiSide);

    static bool overlapCircleVsCircle(
        float xpos1, float ypos1, float radius1,
        float xpos2, float ypos2, float radius2);

    static bool isEmptyRow(
        const Simulation &sim, int xcoord1, int xcoord2, int ycoord, int dir);

    static bool isEmptyColumn(
        const Simulation &sim, int xcoord, int ycoord1, int ycoord2, int dir);

    static int mapVectorToOrientation(float xdir, float ydir);

    static float getTimeOfIntersectionCircleVsCircle(
        float xpos, float ypos, float vx, float vy,
        float a, float b, float radius);

    static float getTimeOfIntersectionCircleVsLineseg(
        float xpos, float ypos, float dx, float dy,
        float a1, float b1, float a2, float b2, float radius);

    static float getTimeOfIntersectionCircleVsArc(
        float xpos, float ypos, float vx, float vy,
        float a, float b, float hor, float ver,
        float radiusArc, float radiusCircle);

    static std::pair<float, float> mapOrientationToVector(int orientation);

    static std::optional<float> getRaycastDistance(
        const Simulation &sim, float xpos, float ypos, float dx, float dy);

    static float intersectRayVsCellContents(
        const Simulation &sim, int xcell, int ycell,
        float xpos, float ypos, float dx, float dy);

    static bool raycastVsPlayer(
        const Simulation &sim, float xstart, float ystart,
        float ninjaXpos, float ninjaYpos, float ninjaRadius);

    static bool checkLinesegVsNinja(
        float x1, float y1, float x2, float y2, const Ninja &ninja);

    static bool overlapCircleVsSegment(
        float xpos, float ypos, float radius,
        float px1, float py1, float px2, float py2);

    static std::optional<std::tuple<bool, std::pair<float, float>>>
    getSingleClosestPoint(const Simulation &sim, float xpos, float ypos, float radius);

private:
    // Cache for frequently used calculations
    static std::unordered_map<std::string, std::vector<std::pair<int, int>>> cellCache;
    static std::unordered_map<float, float> sqrtCache;

    static float getCachedSqrt(float n);
};