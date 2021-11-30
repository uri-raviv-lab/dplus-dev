using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace PDBUnits
{
    class Program
    {
        
        /*static public Matrix3X3 findRotatinMatrix(Vector a, Vector b)
        {

            //Arguments: vector a and vector b. Returns the appropriate R matrix according to the calculation in "Finding Rotation Matrix.doc" (directory Resources) 

            Vector v = Vector.cross(a, b);
            double s = Math.Sqrt(v * v);
            double c = a * b;
            Matrix3X3 Vx = Matrix3X3.SkewSymmetric(v);
            return Matrix3X3.I + Vx + (Vx * Vx) * ((1 - c) / (s * s));
        } */ 
        
        public static void findEulerAngles(Matrix3X3 R)
        {
            double psi1, theta1, phi1, psi2 = -1, theta2 = -1, phi2 = -1;
            if (R != null)
            {
                if (R.matrix[2, 0] != 1 && R.matrix[2, 0] != -1)
                {
                    theta1 = -Math.Asin(R.matrix[2, 0]);
                    theta2 = Math.PI - theta1;
                    psi1 = Math.Atan2(R.matrix[2, 1] / Math.Cos(theta1), R.matrix[2, 2] / Math.Cos(theta1));
                    psi2 = Math.Atan2(R.matrix[2, 1] / Math.Cos(theta2), R.matrix[2, 2] / Math.Cos(theta2));
                    phi1 = Math.Atan2(R.matrix[1, 0] / Math.Cos(theta1), R.matrix[0, 0] / Math.Cos(theta1));
                    phi2 = Math.Atan2(R.matrix[1, 0] / Math.Cos(theta2), R.matrix[0, 0] / Math.Cos(theta2));
                }
                else
                {
                    phi1 = 0;
                    if (R.matrix[2, 0] == -1)
                    {
                        theta1 = Math.PI / 2;
                        psi1 = phi1 + Math.Atan2(R.matrix[0, 1], R.matrix[0, 2]);
                    }
                    else
                    {
                        theta1 = -(Math.PI / 2);
                        psi1 = -(phi1 + Math.Atan2(R.matrix[0, 1], R.matrix[0, 2]));
                    }
                }
                Console.WriteLine("psi1: " + psi1 * 180 / Math.PI + " theta1: " + theta1 * 180 / Math.PI + " phi1: " + phi1 * 180 / Math.PI);
                Console.WriteLine("psi2: " + psi2 * 180 / Math.PI + " theta2: " + theta2 * 180 / Math.PI + " phi1: " + phi2 * 180 / Math.PI);
            }
            else
                Console.WriteLine("Please initialize rotation matrix first");
        }

        static double toRad(double deg)
        {
            return deg * Math.PI / 180;
        }

        static double toDeg(double rad)
        {
            return rad * 180 / Math.PI;
        }

        public static void PrintArray(double[,] a) 
        {
            for (int row = 0; row < a.GetLength(0); row++) 
            {
                Console.Write(a[row, 0]);
                for (int col = 1; col < a.GetLength(1); col++) 
                {
                    Console.Write(", " + a[row, col]);
                }
                Console.Write("\n");
            }
        }

        static void Main(string[] args)
        {
            Atom.initializeAtomicData();

            if(args.Length > 0)
            {
                if(args.Length != 4)
                {
                    PrintUsage();
                    return;
                }

                double cl_RMSD;
                string cl_unit_path = args[0];
                string cl_subUnit_path = args[1];
                string cl_file_path = args[3];


                if (!File.Exists(cl_unit_path))
                {
                    Console.Write("Large unit file was not found (invalid first parameter).");
                    return;
                }

                if (!File.Exists(cl_subUnit_path))
                {
                    Console.Write("Subunit file was not found (invalid second parameter).");
                    return;
                }

                if (!double.TryParse(args[2], out cl_RMSD))
                {
                    Console.Write("Invalid RMSD (third parameter).");
                    return;
                }
                Unit cl_subUnit = new Unit(cl_subUnit_path);
                Unit cl_unit = new Unit(cl_unit_path);

                Unit.createDolFile(cl_unit, cl_subUnit, cl_RMSD, cl_file_path, 0);
                return;
            }

                       
            Console.WriteLine("\n------------------\nDOL FILES CREATOR\n------------------\n");

            Console.WriteLine("Note: Can be run as a command line tool. Run with --help for usage. \n");
            string unit_path, subUnit_path, response;
            double RMSD;
            while(true) 
            {
                Console.Write("Enter the complete structure pdb file path: ");
                unit_path = Console.ReadLine();
                while (!File.Exists(unit_path) || unit_path.IndexOf(".pdb")==-1)
                {
                    Console.Write("File was not found or is not of the pdb format.\nPlease Enter a valid file path: ");
                    unit_path = Console.ReadLine();
                }

                Console.Write("Enter the sub unit pdb file path: ");
                subUnit_path = Console.ReadLine();
                while (!File.Exists(subUnit_path) || subUnit_path.IndexOf(".pdb") == -1)
                {
                    Console.Write("File was not found or is not of the pdb format. Pleas Enter a valid file path: ");
                    subUnit_path = Console.ReadLine();
                }

                Console.Write("Enter Max RMSD: ");
                while (!double.TryParse(Console.ReadLine(), out RMSD))
                {
                    Console.Write("Invalid RMSD. Please try again: ");
                }
                Unit subUnit = new Unit(subUnit_path);
                Unit unit = new Unit(unit_path);

                Console.Write("Enter dol file output path: ");
                string file_path = Console.ReadLine();
                while (File.Exists(file_path))
                {
                    Console.Write("File already exists. Do you wish to overwrite? (y/n): ");
                    response = Console.ReadLine();
                    if (response == "n")
                    {
                        Console.Write("Enter a different path: ");
                        file_path = Console.ReadLine();
                    }
                    else
                    {
                        Console.WriteLine("File will be rewritten");
                        break;
                    }

                }

                Unit.createDolFile(unit, subUnit, RMSD, file_path, 0);
                Console.Write("Do you wish to continue? (y/n): ");
                response = Console.ReadLine();
                if (response == "n")
                    break;
            }
        }

        public static void PrintUsage()
        {

            Console.WriteLine("PDBUnits <Large unit PDB file> <subunit PDB file> <RMSD> <Output DOL file path>");
        }

    }
}
