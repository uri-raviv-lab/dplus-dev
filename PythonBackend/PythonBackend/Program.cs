using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Python.Runtime;
using System.Reflection;

namespace PythonBackend
{
    //class Program
    //{
    //    static void Main(string[] args)
    //    {

    //        var pythonPath = @"C:\Users\yael\Sources\dplus\Python35";

    //        Environment.SetEnvironmentVariable("PATH", $@"{pythonPath};" + Environment.GetEnvironmentVariable("PATH"));

    //        using (Py.GIL())
    //        {
    //            try
    //            {
    //                // add python of the installer to sys path (python should work even if the user dosn't have python on the computer)
    //                dynamic sys = Py.Import("sys");
    //                sys.path.append($@"{pythonPath}\Lib");
    //                sys.path.append(pythonPath);

    //                Console.WriteLine(sys.path);

    //                //generate
    //                Console.WriteLine("Please enter state file name");
    //                string fixed_state_file = System.Console.ReadLine();

    //                dynamic dplus_calc_model = Py.Import("dplus.CalculationRunner");
    //                dynamic CalculationInput_model = Py.Import("dplus.CalculationInput");
    //                //string fixed_state_file = "C:\\Users\\yael\\Sources\\dplus\\PythonInterface\\tests\\reviewer_tests\\files_for_tests\\generate\\cpu\\short\\1Sphere_direct_GK\\1Sphere_direct_GK_fixed.state";
    //                dynamic input = CalculationInput_model.CalculationInput.load_from_state_file(fixed_state_file);
    //                Console.WriteLine(input.use_gpu);
    //                input.use_gpu = false;
    //                dynamic runner = dplus_calc_model.LocalRunner();
    //                dynamic result = runner.generate(input);
    //                Console.WriteLine(result.graph);
    //            }
    //            catch (Exception e)
    //            {
    //                Console.WriteLine("error");
    //                Console.WriteLine(e);
    //            }


    //            //fit python
    //        }

    //    }
    //}
}
