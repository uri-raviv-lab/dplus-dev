using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PDBUnits
{
    class TranslationVector : Vector
    {

        //The class containts three vectors for each axis. each of the vectors represent the translation between two units on each axis

        //First counstuctor is the parent constructor, initializes the three vectors with arguments (See class: Vector)

        public TranslationVector(double x, double y, double z) : base(x, y, z) { }


        static public double calculateRMSD(Unit unit1, Unit unit2, Vector T) 
        {
            //Arguments: two units and T: the vector of the offset between their two center of masses. We are going to try to recreate unit 1 by subtracting T. If the new recreated unit 1 offset from the original unit 1 is around zero, the units are translated

            List<Vector> NewUnit1 = new List<Vector> { }; //We are going to try to recreate Unit 2 atoms cords by T 

            for (int i = 0; i < unit2.atoms.Count; i++)
            {
                NewUnit1.Add(unit2.atoms[i].cords - T);
            }

            double RMSDSum = 0;
            for (int i = 0; i < NewUnit1.Count; i++)
            {
                RMSDSum += (NewUnit1[i] - unit1.atoms[i].cords) ^ 2;
            }
            return Math.Sqrt(RMSDSum / NewUnit1.Count);
        }

        static public Vector findTranslationVector(Unit unit1, Unit unit2, double maxRMSD, EulerAngles e)
        {

            //The function recives two units. If exists, it calculates and initializes the transaltion vectors for the two unit on each axis. returns null if there is no fit.

            if (unit1.atoms.Count > 0 && unit2.atoms.Count > 0)
            {
                unit1.rotate(e.R);
                Vector T = unit2.findCenterOfMass() - unit1.findCenterOfMass();
                double RMSD = TranslationVector.calculateRMSD(unit1, unit2, T);
                if (RMSD <= maxRMSD)
                    return T;
            }
            return null;
        }
    }
}
