using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PDBUnits
{
    class Matrix3X3
    {
        public double[,] matrix = new double[3, 3];

        public static Matrix3X3 I = new Matrix3X3(new double[3, 3] 
        {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        });

        public static Matrix3X3 RotationMatrix(double psi, double theta, double phi)
        {
            /*return new Matrix3X3(new double[,] 
            {
                {Math.Cos(theta)*Math.Cos(phi), Math.Sin(psi)*Math.Sin(theta)*Math.Cos(phi) - Math.Cos(psi)*Math.Sin(phi), Math.Cos(psi)*Math.Sin(theta)*Math.Cos(phi)+Math.Sin(psi)*Math.Sin(phi)},
                {Math.Cos(theta)*Math.Sin(phi), Math.Sin(psi)*Math.Sin(theta)*Math.Sin(phi) + Math.Cos(psi)*Math.Cos(phi), Math.Cos(psi)*Math.Sin(theta)*Math.Sin(phi)-Math.Sin(psi)*Math.Cos(phi)},
                {-Math.Sin(theta), Math.Sin(psi)*Math.Cos(theta),Math.Cos(psi)*Math.Cos(theta)}
            });*/
            return new Matrix3X3(new double[,]
            {
                {Math.Cos(theta)*Math.Cos(phi), -Math.Cos(theta)*Math.Sin(phi), Math.Sin(theta)},
                {Math.Cos(psi)*Math.Sin(phi)+Math.Cos(phi)*Math.Sin(psi)*Math.Sin(theta), Math.Cos(psi)*Math.Cos(phi)-Math.Sin(psi)*Math.Sin(theta)*Math.Sin(phi), -Math.Cos(theta)*Math.Sin(psi)},
                {Math.Sin(psi)*Math.Sin(phi)-Math.Cos(psi)*Math.Cos(phi)*Math.Sin(theta), Math.Cos(phi)*Math.Sin(psi)+Math.Cos(psi)*Math.Sin(theta)*Math.Sin(phi), Math.Cos(psi)*Math.Cos(theta)}
            });
        }
        public Matrix3X3() { }

        public Matrix3X3(double[,] matrix)
        {
            this.matrix = matrix;
        }

        public Matrix3X3(Vector v1, Vector v2, Vector v3)
        {
            this.matrix = new double[,] { { v1.x, v2.x, v3.x }, { v1.y, v2.y, v3.y }, { v1.z, v2.z, v3.z } };
        }
        
        public static Matrix3X3 operator +(Matrix3X3 m1, Matrix3X3 m2) //matrices addition
        {
            Matrix3X3 result = new Matrix3X3(new double[3, 3]);
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    result.matrix[row, col] = m1.matrix[row, col] + m2.matrix[row, col];
                }
            }
            return result;
        }

        public static Matrix3X3 operator -(Matrix3X3 m1, Matrix3X3 m2) //matrices subtraction
        {
            Matrix3X3 result = new Matrix3X3(new double[3, 3]);
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    result.matrix[row, col] = m1.matrix[row, col] - m2.matrix[row, col];
                }
            }
            return result;
        }

        public static Matrix3X3 operator *(Matrix3X3 m, double a) //scalar multiplication
        {
            Matrix3X3 result = new Matrix3X3(new double[3, 3]);
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    result.matrix[row, col] = m.matrix[row, col] * a;
                }
            }
            return result;
        }

        public static Matrix3X3 operator *(Matrix3X3 m1, Matrix3X3 m2) //matrices multiplication
        {
            Matrix3X3 result = new Matrix3X3(new double[3, 3]);
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        result.matrix[row, col] += m1.matrix[row, k] * m2.matrix[k, col];
                    }
                }
            }
            return result;
        }

        public static Vector operator *(Matrix3X3 m, Vector v) 
        {
            return new Vector(m.matrix[0, 0] * v.x + m.matrix[0, 1] * v.y + m.matrix[0, 2] * v.z, m.matrix[1, 0] * v.x + m.matrix[1, 1] * v.y + m.matrix[1, 2] * v.z, m.matrix[2, 0] * v.x + m.matrix[2, 1] * v.y + m.matrix[2, 2] * v.z);
        }

        public static Matrix3X3 SkewSymmetric(Vector v)
        {
            return new Matrix3X3(new double[3, 3] 
            { 
                {0, -v.z, v.y},
                {v.z, 0, -v.x},
                {-v.y, v.x, 0}
            });
        }

        public static double[,] spliceForDeterminant(double[,] source, int rowPos, int colPos)
        {
            double[,] result = new double[source.GetLength(0) - 1, source.GetLength(1) - 1];
            int colFactor = 0, rowFactor = 0;

            for (int row = 0; row < source.GetLength(0); row++)
            {
                if (row != rowPos)
                {
                    colFactor = 0;
                    for (int col = 0; col < source.GetLength(0); col++)
                    {
                        if (col != colPos)
                        {
                            result[row - rowFactor, col - colFactor] = source[row, col];
                        }
                        else
                        {
                            colFactor = 1;
                        } 
                    }
                }
                else
                {
                    rowFactor = 1;
                }
            }
            return result;
        }

        public static double recursiveDeterminant(double[,] matrix)
        {
            if (matrix.GetLength(0) == 2 && matrix.GetLength(1) == 2)
            {
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            }
            double det = 0;
            for (int i = 0; i < matrix.GetLength(1); i++)
            {
                if (i % 2 == 0)
                    det += matrix[0, i] * recursiveDeterminant(spliceForDeterminant(matrix, 0, i));
                else
                    det -= matrix[0, i] * recursiveDeterminant(spliceForDeterminant(matrix, 0, i));
            }
            return det;
        }

        public static double Determinant(double[,] matrix)
        {
            if (matrix.GetLength(0) == 1 && matrix.GetLength(1) == 1)
            {
                return matrix[0, 0];
            }
            else if (matrix.GetLength(0) == matrix.GetLength(1))
            {
                return recursiveDeterminant(matrix);
            }
            else
            {
                return 0;
            }
        }

        public static double[,] Transpose(double[,] m)
        {
            double[,] result = new double[m.GetLength(1), m.GetLength(0)];
            for (int row = 0; row < result.GetLength(0); row++)
                for (int col = 0; col < result.GetLength(1); col++)
                    result[row, col] = m[col, row];
            return result;
        }

        public static double[,] Adjugate(double[,] m)
        {
            double[,] transpose = Matrix3X3.Transpose(m);
            double[,] result = new double[m.GetLength(0), m.GetLength(1)];

            for (int row = 0; row < result.GetLength(0); row++)
            {
                for (int col = 0; col < result.GetLength(1); col++)
                {
                    double sign = Math.Pow(-1, row + col + 2);
                    double[,] detArray = Matrix3X3.spliceForDeterminant(transpose, row, col);
                    double det=Matrix3X3.Determinant(detArray);
                    result[row, col] = sign * det;
                }
            }
            return result;      
        }

        public static double[,] Inverse(double[,] m)
        {
            double[,] result = Adjugate(m);
            double d = 1 / Determinant(m);
            for (int row = 0; row < result.GetLength(0); row++)
                for (int col = 0; col < result.GetLength(1); col++)
                    result[row, col] *= d;
            return result;
        }

        public void PrintMatrix() 
        {
            Console.Write("\n");
            for (int row = 0; row < 3; row++)
            {
                Console.Write("{0, 6}", Math.Round(this.matrix[row, 0], 2));
                for (int col = 1; col < 3; col++)
                {
                    Console.Write("{1,6}", col + 1, Math.Round(this.matrix[row, col], 2));
                }
                Console.Write("\n");
            }
        }
    }
}
