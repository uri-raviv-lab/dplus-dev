using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PDBUnits
{
    class Vector
    {
        public double x, y, z;

        public Vector(double x, double y, double z)
        {

            //initializes the vectors with arguments

            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static Vector rotate(Vector v, double psi, double theta, double phi)
        {
            return Matrix3X3.RotationMatrix(psi, theta, phi) * v;
        }

        public static Vector rotate(Vector v, Matrix3X3 R)
        {
            return R * v;
        }

        public static Vector operator -(Vector v1, Vector v2)
        {
            return new Vector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        }

        public static Vector operator +(Vector v1, Vector v2)
        {
            return new Vector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
        }

        public static double operator ^(Vector v, double a)
        {
            return Math.Pow(v.x, a) + Math.Pow(v.y, a) + Math.Pow(v.z, a);
        }

        public static Vector operator /(Vector v, double a) //scalar division
        {
            return new Vector(v.x / a, v.y / a, v.z / a);
        }

        public static Vector operator *(Vector v, double a) //scalar multiplication
        {
            return new Vector(v.x * a, v.y * a, v.z * a);
        }

        public static double operator *(Vector v1, Vector v2)
        {
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        }

        public static Vector norm(Vector a)
        {
            double N = a * a;
            return a * 1.0 / Math.Sqrt(N);
        }

        public static Vector cross(Vector a, Vector b)
        {
            return new Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        }

        public void PrintVector()
        {
            Console.WriteLine("Vector\n---------------");
            Console.WriteLine("X: " + this.x + " Y: " + this.y + " z: " + this.z);
            Console.WriteLine("---------------");
        }
    }
}
