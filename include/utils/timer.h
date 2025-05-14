#pragma once
#include <chrono>
#include <sstream>
#include <iomanip>

class Timer
{
public:
  void start();
  double ms() const;
  double sec() const;
  double min() const;
  std::string elapsed(int precision = 2) const;

private:
  using clk = std::chrono::high_resolution_clock;
  clk::time_point t0;
};
