#pragma once

using namespace System;

#include "Utility.h"
#include "Point3D.h"

namespace GLView 
{
	public ref class Camera sealed
	{
	// Properties
	public:
		/// <summary>
		/// Gets or sets the camera position.
		/// </summary>		
		property Point3D Position;
		/// <summary>
		/// Gets or sets the camera target.
		/// </summary>		
		property Point3D Target;
		/// <summary>
		/// Gets or sets the camera roll angle in degrees.
		/// </summary>
		property float Roll;
		/// <summary>
		/// Gets the camera up direction.
		/// </summary>
		property Point3D Up
		{
			virtual Point3D get(void) 
			{ 
				float pitch = (this->Pitch + 90) * (float)Math::PI / 180.0f;
				float yaw = this->Yaw * (float)Math::PI / 180.0f;
				float roll = this->Roll * (float)Math::PI / 180.0f;

				// Rotate according to pitch and yaw
				float x = 1.0f * (float)Math::Cos(pitch) * (float)Math::Cos(yaw);
				float y = 1.0f * (float)Math::Cos(pitch) * (float)Math::Sin(yaw);
				float z = 1.0f * (float)Math::Sin(pitch);

				// Rotate along the Forward axis
				Point3D fwd = Forward;
				float u = fwd.X, v = fwd.Y, w = fwd.Z;
				float sroll = (float)Math::Sin(roll), croll = (float)Math::Cos(roll);

				float newx = u*(u*x + v*y + w*z)*(1.0f - croll) + x*croll + (-w*y + v*z) * sroll;
				float newy = v*(u*x + v*y + w*z)*(1.0f - croll) + y*croll + ( w*x - u*z) * sroll;
				float newz = w*(u*x + v*y + w*z)*(1.0f - croll) + z*croll + (-v*x + u*y) * sroll;

				return Point3D(newx, newy, newz);
			}
		}
		/// <summary>
		/// Gets the camera forward direction.
		/// </summary>
		property Point3D Forward
		{
			virtual Point3D get(void) 
			{ 
				float x = Target.X - Position.X;
				float y = Target.Y - Position.Y;
				float z = Target.Z - Position.Z;
				// Return the normalized vector
				return Point3D(x / Distance, y / Distance, z / Distance);
			}
		}
		/// <summary>
		/// Gets the camera right direction.
		/// </summary>
		property Point3D Right
		{
			virtual Point3D get(void) 
			{ 
				Point3D up = Up, fwd = Forward;

				// Compute the cross-product (up x forward) to obtain right
				float x = up.Y * fwd.Z - up.Z * fwd.Y;
				float y = up.Z * fwd.X - up.X * fwd.Z;
				float z = up.X * fwd.Y - up.Y * fwd.X;
				return Point3D(x, y, z);
			}
		}
		/// <summary>
		/// Gets or sets the camera pitch angle in degrees.
		/// </summary>
		property float Pitch
		{
			virtual float get(void) 
			{ 
				double hdist = Math::Sqrt((Position.X - Target.X) * (Position.X - Target.X) + (Position.Y - Target.Y) * (Position.Y - Target.Y));
				float res = (float)(Math::Atan2(Position.Z - Target.Z, hdist) * 180.0 / Math::PI);
				return res;
			}
			virtual void set(float value) 
			{
				UpdatePosition(value, this->Yaw, this->Distance);
			}
		}
		/// <summary>
		/// Gets or sets the camera yaw angle in degrees.
		/// </summary>
		property float Yaw
		{
			virtual float get(void) 
			{ 
				float res = (float)(Math::Atan2(Position.Y - Target.Y, Position.X - Target.X) * 180.0 / Math::PI);
				if(res < 0.0)
					res += 360.0;
				return res;
			}
			virtual void set(float value) 
			{
				UpdatePosition(this->Pitch, value, this->Distance);
			}
		}
		/// <summary>
		/// Gets or sets the distance to camera target.
		/// </summary>
		property float Distance
		{
			virtual float get(void) 
			{ 
				return (float)Math::Sqrt((Position.X - Target.X) * (Position.X - Target.X) + (Position.Y - Target.Y) * (Position.Y - Target.Y) + (Position.Z - Target.Z) * (Position.Z - Target.Z));
			}
			virtual void set(float value) 
			{
				UpdatePosition(this->Pitch, this->Yaw, value);
			}
		}

	// Public methods
	public:
		/// <summary>
		/// Pans the camera.
		/// </summary>
		/// <param name="dx">x offset from current position.</param>
		/// <param name="dy">y offset from current position.</param>
		/// <param name="dz">z offset from current position.</param>
		void Pan(float dx, float dy, float dz)
		{
			if(Target.X != Target.X || Target.Y != Target.Y || Target.Z != Target.Z)
				Target = Point3D(dx, dy, dz);
			else
				Target = Point3D(Target.X + dx, Target.Y + dy, Target.Z + dz);
			if(Position.X != Position.X || Position.Y != Position.Y || Position.Z != Position.Z)
				Position = Point3D(dx, dy, dz);
			else
				Position = Point3D(Position.X + dx, Position.Y + dy, Position.Z + dz);
		}
		/// <summary>
		/// Pans the camera.
		/// </summary>
		/// <param name="oldPosition">old position.</param>
		/// <param name="newPosition">new position.</param>
		void Pan(Point3D oldPosition, Point3D newPosition)
		{
			if(oldPosition.X != oldPosition.X || oldPosition.Y != oldPosition.Y || oldPosition.Z != oldPosition.Z) {
				Pan(newPosition.X, newPosition.Y, newPosition.Z);
				return;
			}
			Pan(newPosition.X - oldPosition.X, newPosition.Y - oldPosition.Y, newPosition.Z - oldPosition.Z);
		}

		virtual System::String ^ ToString() override
		{
			return String::Format("{0} <- {1} (Pitch: {2:0.00} Yaw: {3:0.00} Roll: {4:0.00})", 
				Target, Position, Pitch, Yaw, Roll);
		}

	// Helpers
	protected:
		/// <summary>
		/// Updates the camera position.
		/// </summary>
		/// <param name="pitch">New pitch angle in degrees.</param>
		/// <param name="yaw">New yaw angle in degrees.</param>
		/// <param name="distance">New distance to camera target.</param>
		void UpdatePosition(float pitch, float yaw, float distance)
		{
			if(pitch > 85.0f) pitch = 85.0f;
			if(pitch < -85.0f) pitch = -85.0f;
			pitch = pitch * (float)Math::PI / 180.0f;
			yaw = yaw * (float)Math::PI / 180.0f;
			float x = Target.X + distance * (float)Math::Cos(pitch) * (float)Math::Cos(yaw);
			float y = Target.Y + distance * (float)Math::Cos(pitch) * (float)Math::Sin(yaw);
			float z = Target.Z + distance * (float)Math::Sin(pitch);
			Position = Point3D(x, y, z);
		}

	// Constructor
	private:
		Camera() { }
	public:
		Camera(Point3D position, Point3D target)
		{
			Position = position;
			Target = target;
		}

	// Destructor
	protected:
		~Camera() { }
		
	};
}