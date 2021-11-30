using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PDBUnits
{
    class Atom
    {

        public static Dictionary<string, double> AtomicMasses = new Dictionary<string, double>(); //symbol [string], atomic mass [double]

        public static void initializeAtomicData()
        {
            Atom.initializeAtomicMasses();
        }

        public static void initializeAtomicMasses()
        {
            // Arguments: Atom symbol and cords. The function return an atom object which is initialized with the symbol, cords and the atomic data (mass, charge etc.)
            //The order of the arguments which are being sent to the atom constructor are: symbol, x, y, x, atomic mass

            //The function reads the atom data from a text file called "atomicData.txt". Located in directory "Resources" along the .exe file. Each line in the text file is an atom. the first two chars of each lines (1-2) represents the symbol. Chars 5-11 are the atomic mass. We fit every atom to its line and then we extract the atomic mass and etc.

            if (System.IO.File.Exists(@"Resources\atomicData.txt"))
            {
                string[] lines = System.IO.File.ReadAllLines(@"Resources\atomicData.txt");
                foreach (string line in lines)
                    AtomicMasses.Add(line.Substring(0, 2).Trim().ToUpper(), double.Parse(line.Substring(3)));
            }
        }

        public Vector cords;
        public double atomicMass; 
        public string symbol;

        public Atom(string symbol, double x, double y, double z)
        {

            //The constructor gets four arguments: atom symbol, x, y, z positions and atomic mass. It initializes the class' properties with the arguments.
            this.symbol = symbol.Trim().ToUpper();
            this.cords = new Vector(x, y, z);
            if (Atom.AtomicMasses.ContainsKey(this.symbol))
                this.atomicMass = Atom.AtomicMasses[this.symbol];
            else
                this.atomicMass = -1; //-1 value, the default value is for unknown atomic mass. 
        }

        public Atom(string symbol, Vector cords)
        {

            //The constructor gets four arguments: atom symbol, x, y, z positions and atomic mass. It initializes the class' properties with the arguments.
            this.symbol = symbol.Trim().ToUpper();
            this.cords = cords;
            if (Atom.AtomicMasses.ContainsKey(this.symbol))
                this.atomicMass = Atom.AtomicMasses[this.symbol];
            else
                this.atomicMass = -1; //-1 value, the default value is for unknown atomic mass. 
        }

        public Atom(Atom a)
        {
            //The constructor gets four arguments: atom symbol, x, y, z positions and atomic mass. It initializes the class' properties with the arguments.
            this.symbol = a.symbol.Trim().ToUpper();
            this.cords = a.cords;
            if (Atom.AtomicMasses.ContainsKey(this.symbol))
                this.atomicMass = Atom.AtomicMasses[this.symbol];
            else
                this.atomicMass = -1; //-1 value, the default value is for unknown atomic mass. 
        }
    }
}
