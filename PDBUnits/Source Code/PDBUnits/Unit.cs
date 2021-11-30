using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace PDBUnits
{
    class Unit 
    {
        public List<Atom> atoms = new List<Atom> { }; //List that contains objects of "Atom", each object in the list represents an atom of the unit.

        public Unit() 
        {
            //default constructur, no arguments and no actions
        }

        public Unit(string file_name) 
        {

            //Arguments: the path to the pdb file of the unit. The constructur initializes the list "atoms" by calling to the function,  sending it the path to the pdb

            this.InitializeByPDB(file_name);
        }

        public Unit(List<Atom> atoms) 
        {

            //Arguments: pre-ordered list of "Atom"s objects. The constructur initializes the atoms list with the sent atoms list.

            this.atoms = atoms;
        }

        public static List<Atom> CloneAtomsList(List<Atom> listToClone) 
        {
            List<Atom> newList = new List<Atom> { };
            listToClone.ForEach((item) => 
            {
                newList.Add(new Atom(item));
            });
            return newList;
        }

        public void InitializeByPDB(string file_path)
        {
            
            // Arguments: the path to the pdb file of the unit. Initializes the atoms list according to the pdb file. 
            //The pdb parameters which are taken care of are: Atom symbol, x position, y position and z position. 
            //The parameters are positioned in fixed indexes in the each line of the pdb file. the format can be found here: http://deposit.rcsb.org/adit/docs/pdb_atom_format.html (Section: "Record: Atom")

            if (file_path.IndexOf(".pdb") != -1)
            {
                if (System.IO.File.Exists(file_path))
                {
                    string[] lines = System.IO.File.ReadAllLines(file_path); //lines contains every line of the pdb file.
                    foreach (string line in lines)
                    {
						if (line.Length < 6) continue;
                        if (line.Substring(0, 6).Trim() == "ATOM" || line.Substring(0, 6).Trim() == "HETATM") //pdb line from 1 to 6 is record name. If the record is an atom, we want to add it
                            this.atoms.Add(new Atom(line.Substring(76, 2), double.Parse(line.Substring(30, 8)), double.Parse(line.Substring(38, 8)), double.Parse(line.Substring(46, 8)))); //symbol, x, y, z
                    }
                }
                else
                {
                    Console.WriteLine("File does not exists");
                }
                
            } 
            else
            {
                Console.WriteLine("File type must be pdb. Object was not initialized");
            }
        }
        
        public string adjsutNumberString(string str, int requiredLength) 
        {
            if (str.Length < requiredLength)
            {
                int missing = requiredLength - str.Length;
                if (str.IndexOf(".") == -1)
                {
                    str += ".";
                    missing--;
                }
                for (int i = 1; i <= missing; i++)
                    str += "0";
                return str;
            }
            else if (str.Length > requiredLength)
            {
                str = str.Substring(0, requiredLength);
                if (str[str.Length - 1] == '.')
                    str = str.Substring(0, str.Length - 1);
            }
            return str;
        }

        public string adjustStringRightJustified(string str, int requiredLength)
        {
            string newStr = "";
            if (str.Length < requiredLength)
            {
                int missing = requiredLength - str.Length;
                for (int i = 1; i <= missing; i++)
                    newStr += " ";
                return newStr + str;
            }
            else if (str.Length > requiredLength)
            {
                str = str.Substring(0, requiredLength);
            }
            return str;
        }

        public void exportToPdb()
        {
            Console.Write("Enter pdb file output path: ");
            string file_path = Console.ReadLine();
            while(File.Exists(file_path))
            {
                Console.Write("File already exists. Do you wish to overwrite? (y/n): ");
                string response=Console.ReadLine();
                if(response=="n")
                {
                    Console.Write("Enter a different path: ");
                    file_path = Console.ReadLine();
                }
                else {
                    Console.WriteLine("File will be rewritten");
                    break;
                }
                
            }

            //string file_path = @"F:\PDBUnits\PDB Files\exported.pdb";
            using (StreamWriter file = new StreamWriter(file_path, false))
            {
                for (int i = 0; i < this.atoms.Count; i++)
                {
                    file.WriteLine("ATOM  " + adjustStringRightJustified((i + 1).ToString(), 5) + "                   " + adjsutNumberString(this.atoms[i].cords.x.ToString(), 8) + adjsutNumberString(this.atoms[i].cords.y.ToString(), 8) + adjsutNumberString(this.atoms[i].cords.z.ToString(), 8) + "                      " + adjustStringRightJustified(this.atoms[i].symbol, 2));
                }
            }
            Console.WriteLine("PDB was exported");
        }

        public Vector findCenterOfMass()
        {

            //Arguments: unit which contains list of atoms. The function return the center of mass of the unit

            if (this.atoms.Count > 0)
            {
                Vector Rcm = new Vector(0, 0, 0);
                double masses = 0;

                foreach (Atom atom in this.atoms)
                {
                    if (atom.atomicMass == -1) //-1 value is unknown mass, can't calculate center of mass
                        return null;
                    Rcm += atom.cords * atom.atomicMass;
                    masses += atom.atomicMass;
                }
                return Rcm / masses; //sum of vectors multiplied by masses divided by total mass
            }
            else
                return null;
        }

        public void translate(Vector t, double deviation = 0)
        {
            if (this.atoms.Count > 0)
            {
                for (int i = 0; i < atoms.Count; i++)
                {
                    this.atoms[i].cords += t;
                }
            }
        }

        public void rotate(double psi, double theta, double phi, double deviationPercentage = 0)
        {
            if (this.atoms.Count > 0)
            {
                Vector Rcm = this.findCenterOfMass();
                this.translate(Rcm * -1);
                for (int i = 0; i < atoms.Count; i++)
                {
                    this.atoms[i].cords = Vector.rotate(this.atoms[i].cords, psi, theta, phi);
                }
                this.translate(Rcm);
            }
        }

        public void rotate(Matrix3X3 R, double deviationPercentage = 0)
        {
            if (this.atoms.Count > 0)
            {
                Vector Rcm = this.findCenterOfMass();
                this.translate(Rcm * -1);
                for (int i = 0; i < atoms.Count; i++)
                {
                    this.atoms[i].cords = Vector.rotate(this.atoms[i].cords, R);
                }
                this.translate(Rcm);
            }
        }

        public static bool areSymbolsIdentical(Unit u1, Unit u2)
        {
            if (u1.atoms.Count == u2.atoms.Count && u1.atoms.Count > 0 && u2.atoms.Count > 0)
            {
                for (int i = 0; i < u1.atoms.Count; i++)
                    if (u1.atoms[i].symbol != u2.atoms[i].symbol)
                        return false;
                return true;
            }
            else
            {
                return false;
            }
                
        }

        public void Print()
        {

            //If the atoms list is initialize, the function prints the symbol, x, y and z positions of each atom.
            Console.WriteLine("------------------\nUNIT LIST\n------------------");
            if (this.atoms != null)
            {
                for (int i = 0; i < atoms.Count; i++)
                {
                    Console.WriteLine((i+1)+". ATOM: " + this.atoms[i].symbol + " x: " + this.atoms[i].cords.x + " y: " + this.atoms[i].cords.y + " z: " + this.atoms[i].cords.z);
                }
            }
        }

        public static double radToDOLDisplay(double rad)
        {
            return rad * 180 / Math.PI;
        }

        public Unit Trim(int startPos,int count)
        {
            return new Unit(this.atoms.GetRange(startPos, count));
        }

        public static Unit findSubUnit(Unit unit, Unit subUnit, int startPos)
        {
            int endPos = startPos + 1;
            for (int subInd = 1; (endPos - startPos) < subUnit.atoms.Count; endPos++, subInd++)
                if (endPos >= unit.atoms.Count || unit.atoms[endPos].symbol != subUnit.atoms[subInd].symbol)
                    return null;
            return unit.Trim(startPos, endPos - startPos);
        }

        public static double RoundDOLValue(int decimals, double value)
        {
            if (decimals != 0)
                return Math.Round(value, decimals);
            return value;
        }

        public static void createDolFile(Unit unit, Unit subUnit, double RMSD, string file_path, int decimals) //creates DOL for D+ 
        {

            //string file_path = @"F:\PDBUnits\Paper\Hep\HepDOL.dol";

            using (StreamWriter file = new StreamWriter(file_path, false))
            {
                int dolCounter = 1;
                for (int i = 0; i < unit.atoms.Count; i++)
                {
                    if (unit.atoms[i].symbol == subUnit.atoms[0].symbol)
                    {
                        Unit transformedSubUnit = findSubUnit(unit, subUnit, i);
                        if (transformedSubUnit != null)
                        {
                            Console.WriteLine("SUB UNIT FOUND AT ATOM ID: " + i + "\n------------------");
                            EulerAngles e = EulerAngles.findRotationBetweenUnits(subUnit, transformedSubUnit, RMSD);
                            if (e != null)
                            {
                                Console.WriteLine("Rotation Fit!");
                                e.PrintEulerAnglesDeg();
                                Vector t = TranslationVector.findTranslationVector(new Unit(Unit.CloneAtomsList(subUnit.atoms)), transformedSubUnit, RMSD, e); ///find translation vectors between the vectors
                                if (t != null) //if we found a translation fit in the line above, display the translation. if t (translation vectors) is null, there's no fit and the vectors have no translation fit between them.   
                                {
                                    Console.WriteLine("Traslation Fit! x: " + Math.Round(t.x, 3) + " y: " + Math.Round(t.y, 3) + " z: " + Math.Round(t.z, 3));
                                    file.WriteLine(dolCounter + " " + (Unit.RoundDOLValue(decimals, t.x)) / 10 + " " + (Unit.RoundDOLValue(decimals, t.y)) / 10 + " " + (Unit.RoundDOLValue(decimals, t.z)) / 10 + " " + Unit.RoundDOLValue(decimals, radToDOLDisplay(e.psi1)) + " " + Unit.RoundDOLValue(decimals, radToDOLDisplay(e.theta1)) + " " + Unit.RoundDOLValue(decimals, radToDOLDisplay(e.phi1))); //Cords are divided by 10 to match D+ distance units: Angstroms (PDB) vs. Nanometer (D+). Euler angles are converted to degrees from radians. 
                                    dolCounter++;
                                }
                                else
                                    Console.WriteLine("Translation RMSD exceeded");
                            }
                            else
                            {
                                Console.WriteLine("Rotation RMSD exceeded: ");
                            }
                        }
                    }
                }
            }
            Console.WriteLine("\n------------------\nFile was created\n------------------\n");
        }

        /*public static Unit createUnitByDOL(Unit subUnit, string DOL_path)
        {
            f (System.IO.File.Exists(file_path))
                {
                    string[] lines = System.IO.File.ReadAllLines(file_path); //lines contains every line of the pdb file.
                    foreach (string line in lines)
                    {
        }*/

        public void joinUnit(Unit u)
        {
            for (int i = 0; i < u.atoms.Count; i++)
            {
                this.atoms.Add(u.atoms[i]);
            }
        }

        public static void Unitcheck(Unit p1, Unit p2)
        {
            if (Unit.areSymbolsIdentical(p1, p2))
            {
                EulerAngles e = EulerAngles.findRotationBetweenUnits(p1, p2, 0.00000001);
                if (e != null)
                {
                    Console.WriteLine("Rotation Fit!");
                    e.PrintEulerAnglesDeg();
                    Vector t = TranslationVector.findTranslationVector(p1, p2, 0.000001, e); ///find translation vectors between the vectors
                    if (t != null) //if we found a translation fit in the line above, display the translation. if t (translation vectors) is null, there's no fit and the vectors have no translation fit between them.
                    {
                        Console.WriteLine("Traslation Fit! x: " + Math.Round(t.x, 3) + " y: " + Math.Round(t.y, 3) + " z: " + Math.Round(t.z, 3));
                    }
                    else
                    {
                        Console.WriteLine("No Fit");
                    }
                }
                else
                {
                    Console.WriteLine("No Rotation Fit");
                }
            }
            else
            {
                Console.WriteLine("Symbols are not identical");
            }
        }
    }
}
