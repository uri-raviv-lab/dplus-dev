using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PDBUnits
{
    class Matrix
    {
        //work in progress
        public double[,] matrix;

         public static Matrix I = new Matrix(new double[3, 3] 
        {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        });

        public static Matrix RotationMatrix(double psi, double theta, double phi)
        {
            return new Matrix(new double[,] 
            {
                {Math.Cos(theta)*Math.Cos(phi), Math.Sin(psi)*Math.Sin(theta)*Math.Cos(phi) - Math.Cos(psi)*Math.Sin(phi), Math.Cos(psi)*Math.Sin(theta)*Math.Cos(phi)+Math.Sin(psi)*Math.Sin(phi)},
                {Math.Cos(theta)*Math.Sin(phi), Math.Sin(psi)*Math.Sin(theta)*Math.Sin(phi) + Math.Cos(psi)*Math.Cos(phi), Math.Cos(psi)*Math.Sin(theta)*Math.Sin(phi)-Math.Sin(psi)*Math.Cos(phi)},
                {-Math.Sin(theta), Math.Sin(psi)*Math.Cos(theta),Math.Cos(psi)*Math.Cos(theta)}
            });
        }
        public Matrix() { }

        public Matrix(double[,] matrix)
        {
            this.matrix = matrix;
        }

        public static bool areDimensionsEqual(Matrix m1, Matrix m2)
        {
            if (m1.matrix.GetLength(0) == m2.matrix.GetLength(0) && m1.matrix.GetLength(1) == m2.matrix.GetLength(1))
                return true;
            Console.WriteLine("Matrices dimensions are not equal, cannot execute command");
            return false;
        }

        public static bool multiplicationFit(Matrix m1, Matrix m2)
        {
            if (m1.matrix.GetLength(0) == m2.matrix.GetLength(1) && m1.matrix.GetLength(1) == m2.matrix.GetLength(0))
                return true;
            Console.WriteLine("Matrices are not fit for multiplication, cannot execute command");
            return false;
        }

        public static Matrix operator +(Matrix m1, Matrix m2) //matrices addition
        {
            if (areDimensionsEqual(m1, m2))
            {
                Matrix result = new Matrix(new double[m1.matrix.GetLength(0), m1.matrix.GetLength(1)]);
                for (int row = 0; row < result.matrix.GetLength(0); row++)
                {
                    for (int col = 0; col < result.matrix.GetLength(1); col++)
                    {
                        result.matrix[row, col] = m1.matrix[row, col] + m2.matrix[row, col];
                    }
                }
                return result;
            }
            return null;
        }

        public static Matrix operator -(Matrix m1, Matrix m2) //matrices subtraction
        {
            if (areDimensionsEqual(m1, m2))
            {
                Matrix result = new Matrix(new double[m1.matrix.GetLength(0), m1.matrix.GetLength(1)]);
                for (int row = 0; row < result.matrix.GetLength(0); row++)
                {
                    for (int col = 0; col < result.matrix.GetLength(1); col++)
                    {
                        result.matrix[row, col] = m1.matrix[row, col] - m2.matrix[row, col];
                    }
                }
                return result;
            }
            return null;
        }

        public static Matrix operator *(Matrix m, double a) //scalar multiplication
        {
            for (int row = 0; row < m.matrix.GetLength(0); row++)
            {
                for (int col = 0; col < m.matrix.GetLength(1); col++)
                {
                    m.matrix[row, col] *= a;
                }
            }
            return m;
        }

        public static Matrix operator *(Matrix m1, Matrix m2) //matrices multiplication
        {
            if (multiplicationFit(m1, m2))
            {
                Matrix result = new Matrix(new double[m1.matrix.GetLength(0), m2.matrix.GetLength(1)]);
                for (int row = 0; row < result.matrix.GetLength(0); row++)
                {
                    for (int col = 0; col < result.matrix.GetLength(1); col++)
                    {
                        for (int k = 0; k < m1.matrix.GetLength(1); k++)
                        {
                            result.matrix[row, col] += m1.matrix[row, k] * m2.matrix[k, col];
                        }
                    }
                }
                return result;
            }
            return null;
        }

        /*public static Vector operator *(Matrix3X3 m, Vector v) 
        {
            return new Vector(m.matrix[0, 0] * v.x + m.matrix[0, 1] * v.y + m.matrix[0, 2] * v.z, m.matrix[1, 0] * v.x + m.matrix[1, 1] * v.y + m.matrix[1, 2] * v.z, m.matrix[2, 0] * v.x + m.matrix[2, 1] * v.y + m.matrix[2, 2] * v.z);
        }*/

        public static Matrix SkewSymmetric(Vector v)
        {
            return new Matrix(new double[3, 3] 
            { 
                {0, -v.z, v.y},
                {v.z, 0, -v.x},
                {-v.y, v.x, 0}
            });
        }

        public static Matrix spliceForDeterminant(double[,] source, int rowPos, int colPos)
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
            return new Matrix(result);
        }

        public static double recursiveDeterminant(Matrix m)
        {
            if (m.matrix.GetLength(0) == 2 && m.matrix.GetLength(1) == 2)
            {
                return m.matrix[0, 0] * m.matrix[1, 1] - m.matrix[0, 1] * m.matrix[1, 0];
            }
            double det = 0;
            for (int i = 0; i < m.matrix.GetLength(1); i++)
            {
                if (i % 2 == 0)
                    det += m.matrix[0, i] * recursiveDeterminant(spliceForDeterminant(m.matrix, 0, i));
                else
                    det -= m.matrix[0, i] * recursiveDeterminant(spliceForDeterminant(m.matrix, 0, i));
            }
            return det;
        }

        public static double Determinant(Matrix m)
        {
            if (m.matrix.GetLength(0) == 1 && m.matrix.GetLength(1) == 1)
            {
                return m.matrix[0, 0];
            }
            else if (m.matrix.GetLength(0) == m.matrix.GetLength(1))
            {
                return recursiveDeterminant(m);
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
                    double sign = Math.Pow(-1, row + col + 2); //sign (+ or -) according to current position (row, col)
                    double det = Matrix3X3.Determinant(Matrix3X3.spliceForDeterminant(transpose, row, col)); //matrix transpose determinant
                    result[row, col] = sign * det; 
                }
            }
            return result;      
        }

        public static double[,] Inverse(double[,] m)
        {
            double[,] result = Adjugate(m);
            /*double d = 1 / Determinant(m);
            for (int row = 0; row < result.GetLength(0); row++)
                for (int col = 0; col < result.GetLength(1); col++)
                    result[row, col] *= d;*/
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
