#pragma once

using namespace System;

namespace GLView {

	/// <summary>
	/// Represents a performance timer with 1 ms resolution.
	/// </summary>
	ref class GLPerformanceTimer
	{
	private:
		unsigned long mStartTime, mStopTime, mDelta;
		bool mStarted;

	public:
		GLPerformanceTimer()
		{
			mStarted = false;
			Start();
			mDelta = Stop();
		}

		GLPerformanceTimer(bool StartTimer)
		{
			mStarted = false;
			Start();
			mDelta = Stop();
			if (StartTimer) Start();
		}

	protected:
		~GLPerformanceTimer() { }

	public:
		inline void Start(void) 
		{
			mStartTime = GetTickCount();
			mStarted = true;
		}

		inline unsigned long Stop() 
		{
			if (!mStarted) throw gcnew Exception(L"Timer is not running. GLPerformanceTimer.Stop() called before GLPerformanceTimer.Start().");
			mStarted = false;
			mStopTime = GetTickCount();
			unsigned long mDuration = (mStopTime - mStartTime);
			if (mDuration < mDelta)
				return 0;
			else
				return (mDuration - mDelta);
		}
	};

}