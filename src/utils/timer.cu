#include "utils/timer.h"

void Timer::start()
{
  t0 = clk::now();
}

double Timer::ms() const
{
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

double Timer::sec() const
{
  return std::chrono::duration<double>(clk::now() - t0).count();
}

double Timer::min() const
{
  return std::chrono::duration<double, std::ratio<60>>(clk::now() - t0).count();
}

std::string Timer::elapsed(int precision) const
{
  auto duration = clk::now() - t0;
  auto seconds = std::chrono::duration<double>(duration).count();

  std::stringstream result;
  result << std::fixed << std::setprecision(precision);

  if (seconds < 1.0)
  {
    result << std::chrono::duration<double, std::milli>(duration).count() << " ms";
  }
  else if (seconds < 60.0)
  {
    result << seconds << " sec";
  }
  else
  {
    result << std::chrono::duration<double, std::ratio<60>>(duration).count() << " min";
  }

  return result.str();
}
