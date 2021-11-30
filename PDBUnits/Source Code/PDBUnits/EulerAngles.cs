using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PDBUnits
{
    class EulerAngles
    {
        public double psi1, theta1, phi1, theta2, psi2, phi2, RMSD;
        public Matrix3X3 R;

        public static EulerAngles eI = new EulerAngles(Matrix3X3.I, 0, 0, 0, 0);
        public EulerAngles() { }

        public EulerAngles(Matrix3X3 R, double psi, double theta, double phi, double RMSD) 
        {
            this.R = R;
            this.psi1 = psi;
            this.theta1 = theta;
            this.phi1 = phi;
            this.RMSD = RMSD;
        }
        public EulerAngles(Matrix3X3 R, double psi1, double theta1, double phi1, double psi2, double theta2, double phi2, double RMSD)
        {
            this.R = R;
            this.psi1 = psi1;
            this.theta1 = theta1;
            this.phi1 = phi1;
            this.psi2 = psi2;
            this.theta2 = theta2;
            this.phi2 = phi2;
            this.RMSD = RMSD;
        }

        public EulerAngles(double psi, double theta, double phi, double RMSD)
        {
            this.psi1 = psi;
            this.theta1 = theta;
            this.phi1 = phi;
            this.RMSD = RMSD;
        }

        public void findRotationMatrix(Vector a, Vector b)
        {

            //Arguments: vector a and vector b. Returns the appropriate R matrix according to the calculation in "Finding Rotation Matrix.doc" (directory Resources) 

            Vector v = Vector.cross(a, b);
            double s = Math.Sqrt(v * v);
            double c = a * b;
            Matrix3X3 Vx = Matrix3X3.SkewSymmetric(v);
            this.R = Matrix3X3.I + Vx + (Vx * Vx) * ((1 - c) / (s * s));
        }

        public static Matrix3X3 findRotationMatrix(Matrix3X3 Ui, Matrix3X3 Uf)
        {
            return Uf * new Matrix3X3(Matrix3X3.Inverse(Ui.matrix));
        }

        public static EulerAngles findEulerAngles(Matrix3X3 R, double RMSD)
        {
            if (R != null)
            {
                // Translated from Eigen's eulerAngles, which is taken from Graphics Gems IV
                /*
                const int a0 = 0, a1 = 1, a2 = 2;
                const int odd = ((a0 + 1) % 3 == a1) ? 0 : 1; // == 0
                const int i = a0; // == 0
                const int j = (a0 + 1 + odd) % 3; // == 1
                const int k = (a0 + 2 - odd) % 3; // == 2
                */
                double psi = 0, theta = 0, phi = 0;

                psi = Math.Atan2(R.matrix[1, 2], R.matrix[2, 2]);
                double c2 = Math.Sqrt(R.matrix[0,0]*R.matrix[0,0] + R.matrix[0,1]*R.matrix[0,1]);
                if (psi > 0)
                {
                    psi = psi - Math.PI;
                    theta = Math.Atan2(-R.matrix[0, 2], -c2);
                }
                else
                    theta = Math.Atan2(-R.matrix[0, 2], c2);
                double s1 = Math.Sin(psi);
                double c1 = Math.Cos(psi);
                phi = Math.Atan2(s1 * R.matrix[2, 0] - c1 * R.matrix[1, 0], c1 * R.matrix[1, 1] - s1 * R.matrix[2, 1]);

                //if (!odd) // true
                //res = -res;
                phi = -phi;
                psi = -psi;
                theta = -theta;

                return new EulerAngles(R, psi, theta, phi, RMSD);
            }
            return null;
        }

        public static double calculateRMSD(Unit unit1, Unit unit2, Matrix3X3 R)
        {
            List<Vector> NewUnit2 = new List<Vector> { };

            for (int i = 0; i < unit1.atoms.Count; i++)
            {
                NewUnit2.Add(Vector.rotate(unit1.atoms[i].cords, R));
            }
            
            double RMSDSum = 0;
            for (int i = 0; i < NewUnit2.Count; i++)
            {
                RMSDSum += (NewUnit2[i] - unit2.atoms[i].cords) ^ 2;
            }
            return Math.Sqrt(RMSDSum / NewUnit2.Count);
        }

        public static Matrix3X3 findPrincipalAxis(Unit u) 
        {
            return new Matrix3X3(u.atoms[1].cords - u.atoms[0].cords, u.atoms[2].cords - u.atoms[0].cords, u.atoms[3].cords - u.atoms[0].cords);
        }

        public static EulerAngles findRotationBetweenUnits(Unit unit1, Unit unit2, double maxRMSD)
        {
            if (unit1.atoms.Count > 4 && unit2.atoms.Count > 4)
            {  
                Vector Rcm1 = unit1.findCenterOfMass(), Rcm2 = unit2.findCenterOfMass();
                unit1.translate(Rcm1 * -1);
                unit2.translate(Rcm2 * -1);
                Matrix3X3 R = findRotationMatrix(findPrincipalAxis(unit1), findPrincipalAxis(unit2));
                double RMSD = EulerAngles.calculateRMSD(unit1, unit2, R);
                unit1.translate(Rcm1);
                unit2.translate(Rcm2);
                if (RMSD <= maxRMSD)
                    return EulerAngles.findEulerAngles(R, RMSD);
                Console.WriteLine("RMSD exceeded: "+RMSD);
                return null;
            }
            else
            {
                Console.WriteLine("Unit is too small");
                return null;
            }
        }

        public void PrintEulerAngles()
        {
            Console.WriteLine("Euler Angles\n---------------");
            Console.WriteLine("psi1: " + psi1 * 180 / Math.PI + " theta1: " + theta1 * 180 / Math.PI + " phi1: " + phi1 * 180 / Math.PI);
            Console.WriteLine("psi2: " + psi2 * 180 / Math.PI + " theta2: " + theta2 * 180 / Math.PI + " phi1: " + phi2 * 180 / Math.PI);
            Console.WriteLine("---------------");
        }

        public void PrintEulerAnglesDeg()
        {
            Console.WriteLine("Euler Angles\n---------------");
            Console.WriteLine("psi1: " + psi1 * 180 / Math.PI + " theta1: " + theta1 * 180 / Math.PI + " phi1: " + phi1 * 180 / Math.PI);
            Console.WriteLine("psi2: " + psi2 * 180 / Math.PI + " theta2: " + theta2 * 180 / Math.PI + " phi1: " + phi2 * 180 / Math.PI);
            Console.WriteLine("RMSD: " + RMSD);
            Console.WriteLine("---------------");
        }

        public static EulerAngles operator *(EulerAngles e, double a)
        {
            e.psi1 *= a;
            e.theta1 *= a;
            e.phi1 *= a;
            return e;
        }
    }
}
