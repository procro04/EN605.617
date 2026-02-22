#pragma once

#include <chrono>
#include <iostream>
#include <stdexcept>

class CodeTimer
{
public:
    CodeTimer()
        : m_running(false)
        , m_startTime{}
        , m_endTime{}
    {}

    void startTiming()
    {
        if (m_running)
            throw std::runtime_error("TimingClass: Timer is already running. Call stopTiming() first.");

        m_running   = true;
        m_startTime = std::chrono::high_resolution_clock::now();
    }

    void stopTiming()
    {
        if (!m_running)
            throw std::runtime_error("TimingClass: Timer is not running. Call startTiming() first.");

        m_endTime = std::chrono::high_resolution_clock::now();
        m_running = false;
    }

    void timingResults() const
    {
        if (m_running)
            throw std::runtime_error("TimingClass: Timer is still running. Call stopTiming() first.");

        // Duration as floating-point seconds — fractional seconds are preserved
        const std::chrono::duration<double> elapsed = m_endTime - m_startTime;

        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    }

    // Convenience method if you just want the raw value back
    double elapsedSeconds() const
    {
        if (m_running)
            throw std::runtime_error("TimingClass: Timer is still running. Call stopTiming() first.");

        const std::chrono::duration<double> elapsed = m_endTime - m_startTime;
        return elapsed.count();
    }

private:
    bool                                                     m_running;
    std::chrono::high_resolution_clock::time_point          m_startTime;
    std::chrono::high_resolution_clock::time_point          m_endTime;
};
