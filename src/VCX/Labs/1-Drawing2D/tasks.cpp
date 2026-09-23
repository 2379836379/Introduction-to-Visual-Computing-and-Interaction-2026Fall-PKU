#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <spdlog/spdlog.h>

#include "Labs/1-Drawing2D/tasks.h"

using VCX::Labs::Common::ImageRGB;

namespace VCX::Labs::Drawing2D {
    /******************* 1.Image Dithering *****************/
    void DitheringThreshold(
        ImageRGB &       output,
        ImageRGB const & input) {
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                output.At(x, y) = {
                    color.r > 0.5 ? 1 : 0,
                    color.g > 0.5 ? 1 : 0,
                    color.b > 0.5 ? 1 : 0,
                };
            }
    }

    void DitheringRandomUniform(
        ImageRGB &       output,
        ImageRGB const & input) {
        // A fixed seed keeps the result reproducible while retaining the
        // intended independent, uniformly distributed perturbation.
        std::mt19937 generator(0x2D2D2D2D);
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
        for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
            for (std::size_t x = 0; x < input.GetSizeX(); ++x) {
                glm::vec3 color = input.At(x, y) + glm::vec3(distribution(generator));
                output.At(x, y) = {
                    color.r > 0.5f ? 1.f : 0.f,
                    color.g > 0.5f ? 1.f : 0.f,
                    color.b > 0.5f ? 1.f : 0.f,
                };
            }
        }
    }

    void DitheringRandomBlueNoise(
        ImageRGB &       output,
        ImageRGB const & input,
        ImageRGB const & noise) {
        if (noise.GetSizeX() == 0 || noise.GetSizeY() == 0) {
            DitheringThreshold(output, input);
            return;
        }
        for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
            for (std::size_t x = 0; x < input.GetSizeX(); ++x) {
                auto const noiseX = x % noise.GetSizeX();
                auto const noiseY = y % noise.GetSizeY();
                // Blue-noise textures are encoded in [0, 1], so recenter
                // them before using them as a threshold perturbation.
                glm::vec3 color = input.At(x, y) + noise.At(noiseX, noiseY) - glm::vec3(0.5f);
                output.At(x, y) = {
                    color.r > 0.5f ? 1.f : 0.f,
                    color.g > 0.5f ? 1.f : 0.f,
                    color.b > 0.5f ? 1.f : 0.f,
                };
            }
        }
    }

    void DitheringOrdered(
        ImageRGB &       output,
        ImageRGB const & input) {
        // A dispersed 3x3 threshold pattern.  The half-step in the
        // threshold makes each of the nine levels occupy an equal interval.
        static constexpr int matrix[3][3] {
            { 0, 7, 3 },
            { 6, 5, 2 },
            { 4, 1, 8 },
        };
        for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
            for (std::size_t x = 0; x < input.GetSizeX(); ++x) {
                glm::vec3 const color = input.At(x, y);
                for (std::size_t dy = 0; dy < 3; ++dy) {
                    for (std::size_t dx = 0; dx < 3; ++dx) {
                        float const threshold = (float(matrix[dy][dx]) + 0.5f) / 9.f;
                        output.At(3 * x + dx, 3 * y + dy) = {
                            color.r >= threshold ? 1.f : 0.f,
                            color.g >= threshold ? 1.f : 0.f,
                            color.b >= threshold ? 1.f : 0.f,
                        };
                    }
                }
            }
        }
    }

    void DitheringErrorDiffuse(
        ImageRGB &       output,
        ImageRGB const & input) {
        auto const width  = input.GetSizeX();
        auto const height = input.GetSizeY();
        std::vector<glm::vec3> work(width * height);
        for (std::size_t y = 0; y < height; ++y)
            for (std::size_t x = 0; x < width; ++x)
                work[y * width + x] = input.At(x, y);

        auto addError = [&](std::size_t x, std::size_t y, glm::vec3 const error, float weight) {
            work[y * width + x] += error * weight;
        };
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                glm::vec3 const oldColor = work[y * width + x];
                glm::vec3 const quantized {
                    oldColor.r >= 0.5f ? 1.f : 0.f,
                    oldColor.g >= 0.5f ? 1.f : 0.f,
                    oldColor.b >= 0.5f ? 1.f : 0.f,
                };
                output.At(x, y) = quantized;
                glm::vec3 const error = oldColor - quantized;
                if (x + 1 < width) addError(x + 1, y, error, 7.f / 16.f);
                if (y + 1 < height) {
                    if (x > 0) addError(x - 1, y + 1, error, 3.f / 16.f);
                    addError(x, y + 1, error, 5.f / 16.f);
                    if (x + 1 < width) addError(x + 1, y + 1, error, 1.f / 16.f);
                }
            }
        }
    }

    /******************* 2.Image Filtering *****************/
    void Blur(
        ImageRGB &       output,
        ImageRGB const & input) {
        auto const width  = input.GetSizeX();
        auto const height = input.GetSizeY();
        if (width == 0 || height == 0) return;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                glm::vec3 sum(0.f);
                for (int dy = -1; dy <= 1; ++dy) {
                    auto const yy = std::clamp<int>(int(y) + dy, 0, int(height - 1));
                    for (int dx = -1; dx <= 1; ++dx) {
                        auto const xx = std::clamp<int>(int(x) + dx, 0, int(width - 1));
                        sum += input.At(std::size_t(xx), std::size_t(yy));
                    }
                }
                output.At(x, y) = sum / 9.f;
            }
        }
    }

    void Edge(
        ImageRGB &       output,
        ImageRGB const & input) {
        auto const width  = input.GetSizeX();
        auto const height = input.GetSizeY();
        if (width == 0 || height == 0) return;
        // 8-neighbour Laplacian, with negative responses suppressed.
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                glm::vec3 value(0.f);
                for (int dy = -1; dy <= 1; ++dy) {
                    auto const yy = std::clamp<int>(int(y) + dy, 0, int(height - 1));
                    for (int dx = -1; dx <= 1; ++dx) {
                        auto const xx = std::clamp<int>(int(x) + dx, 0, int(width - 1));
                        float const weight = (dx == 0 && dy == 0) ? 8.f : -1.f;
                        value += input.At(std::size_t(xx), std::size_t(yy)) * weight;
                    }
                }
                output.At(x, y) = glm::max(value, glm::vec3(0.f));
            }
        }
    }

    /******************* 3. Image Inpainting *****************/
    void Inpainting(
        ImageRGB &         output,
        ImageRGB const &   inputBack,
        ImageRGB const &   inputFront,
        const glm::ivec2 & offset) {
        output             = inputBack;
        std::size_t width  = inputFront.GetSizeX();
        std::size_t height = inputFront.GetSizeY();
        std::vector<glm::vec3> g(width * height, glm::vec3(0.f));
        auto const backValue = [&](std::size_t x, std::size_t y) {
            auto const bx = static_cast<std::int64_t>(offset.x) + static_cast<std::int64_t>(x);
            auto const by = static_cast<std::int64_t>(offset.y) + static_cast<std::int64_t>(y);
            if (bx < 0 || by < 0 || bx >= static_cast<std::int64_t>(inputBack.GetSizeX()) || by >= static_cast<std::int64_t>(inputBack.GetSizeY()))
                return glm::vec3(0.f);
            return inputBack.At(static_cast<std::size_t>(bx), static_cast<std::size_t>(by));
        };
        auto const boundaryValue = [&](std::size_t x, std::size_t y) {
            return backValue(x, y) - inputFront.At(x, y);
        };
        // set boundary condition
        if (width != 0 && height != 0) {
            for (std::size_t y = 0; y < height; ++y) {
                g[y * width] = boundaryValue(0, y);
                g[y * width + width - 1] = boundaryValue(width - 1, y);
            }
            for (std::size_t x = 0; x < width; ++x) {
                g[x] = boundaryValue(x, 0);
                g[(height - 1) * width + x] = boundaryValue(x, height - 1);
            }
        }

        // Jacobi iteration, solve Ag = b
        for (int iter = 0; iter < 8000; ++iter) {
            for (std::size_t y = 1; y < height - 1; ++y)
                for (std::size_t x = 1; x < width - 1; ++x) {
                    g[y * width + x] = (g[(y - 1) * width + x] + g[(y + 1) * width + x] + g[y * width + x - 1] + g[y * width + x + 1]);
                    g[y * width + x] = g[y * width + x] * glm::vec3(0.25f);
                }
        }

        for (std::size_t y = 0; y < inputFront.GetSizeY(); ++y)
            for (std::size_t x = 0; x < inputFront.GetSizeX(); ++x) {
                auto const bx = static_cast<std::int64_t>(offset.x) + static_cast<std::int64_t>(x);
                auto const by = static_cast<std::int64_t>(offset.y) + static_cast<std::int64_t>(y);
                if (bx < 0 || by < 0 || bx >= static_cast<std::int64_t>(output.GetSizeX()) || by >= static_cast<std::int64_t>(output.GetSizeY())) continue;
                glm::vec3 color = g[y * width + x] + inputFront.At(x, y);
                output.At(static_cast<std::size_t>(bx), static_cast<std::size_t>(by)) = color;
            }
    }

    /******************* 4. Line Drawing *****************/
    void DrawLine(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1) {
        auto const width  = canvas.GetSizeX();
        auto const height = canvas.GetSizeY();
        if (width == 0 || height == 0) return;
        int x = p0.x;
        int y = p0.y;
        int const dx = std::abs(p1.x - p0.x);
        int const sx = p0.x < p1.x ? 1 : -1;
        int const dy = -std::abs(p1.y - p0.y);
        int const sy = p0.y < p1.y ? 1 : -1;
        int err = dx + dy;
        while (true) {
            if (x >= 0 && y >= 0 && x < static_cast<int>(width) && y < static_cast<int>(height))
                canvas.At(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = color;
            if (x == p1.x && y == p1.y) break;
            int const twiceError = 2 * err;
            if (twiceError >= dy) {
                err += dy;
                x += sx;
            }
            if (twiceError <= dx) {
                err += dx;
                y += sy;
            }
        }
    }

    /******************* 5. Triangle Drawing *****************/
    void DrawTriangleFilled(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1,
        glm::ivec2 const p2) {
        auto const width  = canvas.GetSizeX();
        auto const height = canvas.GetSizeY();
        if (width == 0 || height == 0) return;
        auto const edge = [](glm::ivec2 const a, glm::ivec2 const b, std::int64_t x, std::int64_t y) {
            return (x - a.x) * static_cast<std::int64_t>(b.y - a.y) - (y - a.y) * static_cast<std::int64_t>(b.x - a.x);
        };
        auto const area = edge(p0, p1, p2.x, p2.y);
        if (area == 0) {
            DrawLine(canvas, color, p0, p1);
            DrawLine(canvas, color, p1, p2);
            DrawLine(canvas, color, p2, p0);
            return;
        }
        int const minX = std::max(0, std::min({ p0.x, p1.x, p2.x }));
        int const maxX = std::min(static_cast<int>(width) - 1, std::max({ p0.x, p1.x, p2.x }));
        int const minY = std::max(0, std::min({ p0.y, p1.y, p2.y }));
        int const maxY = std::min(static_cast<int>(height) - 1, std::max({ p0.y, p1.y, p2.y }));
        if (minX > maxX || minY > maxY) return;
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                auto const w0 = edge(p1, p2, x, y);
                auto const w1 = edge(p2, p0, x, y);
                auto const w2 = edge(p0, p1, x, y);
                if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0))
                    canvas.At(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = color;
            }
        }
    }

    /******************* 6. Image Supersampling *****************/
    void Supersample(
        ImageRGB &       output,
        ImageRGB const & input,
        int              rate) {
        auto const outputWidth  = output.GetSizeX();
        auto const outputHeight = output.GetSizeY();
        auto const inputWidth   = input.GetSizeX();
        auto const inputHeight  = input.GetSizeY();
        if (outputWidth == 0 || outputHeight == 0 || inputWidth == 0 || inputHeight == 0) return;
        rate = std::max(rate, 1);

        auto sample = [&](float x, float y) {
            x = std::clamp(x, 0.f, float(inputWidth - 1));
            y = std::clamp(y, 0.f, float(inputHeight - 1));
            auto const x0 = static_cast<std::size_t>(std::floor(x));
            auto const y0 = static_cast<std::size_t>(std::floor(y));
            auto const x1 = std::min(x0 + 1, inputWidth - 1);
            auto const y1 = std::min(y0 + 1, inputHeight - 1);
            float const tx = x - float(x0);
            float const ty = y - float(y0);
            glm::vec3 const c00 = input.At(x0, y0);
            glm::vec3 const c10 = input.At(x1, y0);
            glm::vec3 const c01 = input.At(x0, y1);
            glm::vec3 const c11 = input.At(x1, y1);
            return c00 * ((1.f - tx) * (1.f - ty)) + c10 * (tx * (1.f - ty)) + c01 * ((1.f - tx) * ty) + c11 * (tx * ty);
        };
        float const scaleX = float(inputWidth) / float(outputWidth);
        float const scaleY = float(inputHeight) / float(outputHeight);
        float const sampleCount = float(rate * rate);
        for (std::size_t y = 0; y < outputHeight; ++y) {
            for (std::size_t x = 0; x < outputWidth; ++x) {
                glm::vec3 sum(0.f);
                for (int sy = 0; sy < rate; ++sy) {
                    for (int sx = 0; sx < rate; ++sx) {
                        float const sampleX = (float(x) + (float(sx) + 0.5f) / float(rate)) * scaleX - 0.5f;
                        float const sampleY = (float(y) + (float(sy) + 0.5f) / float(rate)) * scaleY - 0.5f;
                        sum += sample(sampleX, sampleY);
                    }
                }
                output.At(x, y) = sum / sampleCount;
            }
        }
    }

    /******************* 7. Bezier Curve *****************/
    // Note: Please finish the function [DrawLine] before trying this part.
    glm::vec2 CalculateBezierPoint(
        std::span<glm::vec2> points,
        float const          t) {
        if (points.empty()) return glm::vec2(0.f);
        std::vector<glm::vec2> work(points.begin(), points.end());
        for (std::size_t level = 1; level < work.size(); ++level) {
            for (std::size_t i = 0; i + level < work.size(); ++i)
                work[i] = (1.f - t) * work[i] + t * work[i + 1];
        }
        return work.front();
    }
} // namespace VCX::Labs::Drawing2D
