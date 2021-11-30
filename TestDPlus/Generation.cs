using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TestStack.White;
using System.Threading;
using TestStack.White.UIItems;
using TestStack.White.UIItems.ListBoxItems;
using TestStack.White.UIItems.Finders;
using TestStack.White.UIItems.WindowItems;
using System.Diagnostics;
using TestStack.White.WindowsAPI;
using System.Linq;
using System.IO;
using System.Windows.Automation;
using System.Collections.Generic;

namespace TestDPlus
{
    [TestClass]
    public class Generation: DPlusCaller
    {
       const double tolerance = 0.9; //tolerance as percentage - if all the values in the compared output files are within 9% of eachother, then the test passes.

       private void OpenStateFile(Application app, Window wnd, string statePathname)
        {
            var importMenuItem = wnd.MenuBar.MenuItem("File", "Import All Parameters...");
            importMenuItem.Click();

            Thread.Sleep(1000); // Wait for window to open
            Window openDialog = app.GetWindows().Where(w => w.Name == "Import State...").First();

            TextBox filenameField = openDialog.Get<TextBox>(SearchCriteria.ByControlType(ControlType.Edit).AndByText("File name:"));
            filenameField.Text = statePathname;

            Button openButton = openDialog.Get<Button>(SearchCriteria.ByControlType(ControlType.Button).AndByText("Open"));
            openButton.Click();
        }

        private void DoGenerate(Window wnd)
        {
            var generateButton = wnd.Get<Button>("generateButton");
            generateButton.Click();

            do
            {
                Thread.Sleep(1000);
                try
                {
                    var progressBar = wnd.Get<ProgressBar>("progressBar");
                }
                catch(AutomationException)
                {
                    break; // Progress bar is gone
                }
            } while (true);
        }

        private void SaveOutput(Application app, Window wnd, string outputPathname)
        {
            var export = wnd.MenuBar.MenuItem("File", "Export 1D Graph...");
            export.Click();

            Thread.Sleep(1000); // Wait for window to open
            Window openDialog = app.GetWindows().Where(w => w.Name == "Save Model As...").First();
            TextBox filenameField = openDialog.Get<TextBox>(SearchCriteria.ByControlType(ControlType.Edit).AndByText("File name:"));
            filenameField.Text = outputPathname;

            Button saveButton = openDialog.Get<Button>(SearchCriteria.ByControlType(ControlType.Button).AndByText("Save"));
            saveButton.Click();
        }

        static List<Tuple<double, double>> ExtractCoordinates(string fileName)
        {

            //create the list to be filled
            List<Tuple<double, double>> coordinateList = new List<Tuple<double, double>>();
            Trace.WriteLine(string.Format("ExtractCoordinates from {0} ...", fileName));
            try
            {
                // Create an instance of StreamReader to read the file. 
                // The using statement also closes the StreamReader :-)
                using (StreamReader reader = new StreamReader(fileName))
                {

                    string line;
                    double x, y;
                    string delimStr = " \t"; //allow delimeter to be space OR tab
                    char[] delimiter = delimStr.ToCharArray();
                    string[] nums = null;


                    while ((line = reader.ReadLine()) != null)
                    {

                        if (line.StartsWith("#") || line.Length == 0)
                        {
                            //skipping comments and blank lines
                            continue;
                        }

                        //parse the line into two strings
                        nums = line.Split(delimiter, 2);

                        //turn each string into a double
                        double.TryParse(nums[0], out x);
                        double.TryParse(nums[1], out y);

                        //create the tuple
                        var coordinates = Tuple.Create(x, y);
                        //add tuple to list    
                        coordinateList.Add(coordinates);
                    }

                }
            }
            catch (Exception e)
            {
                Trace.WriteLine(e.ToString());
            }

            //return the list
            return coordinateList;
        }

        private bool CloseEnough(double value1, double value2, double tolerance)
        {
            double ratio = 0.0;

            if (value1 > value2)
            {
                ratio = value1 / value2;
            }
            else
            {
                ratio = value2 / value1;

            }

            if (ratio - 1.0 > tolerance)
            {
                Trace.WriteLine(string.Format("ratio {0} greater than percentage tolerance {1}", ratio, tolerance));
                return false;
            }
            return true;
        }

        private bool IdenticalFiles(string actualOutputPathname, string expectedOutputPathname, double tolerance)
        {
            Trace.WriteLine("IdenticalFiles called");

            List<Tuple<double, double>> actuals = ExtractCoordinates(actualOutputPathname);
            List<Tuple<double, double>> expecteds = ExtractCoordinates(expectedOutputPathname);

            if (actuals.Count != expecteds.Count)
            {
                return false;
            }

            for (int i = 0; i < actuals.Count; i++ ) // Loop through both lists with index
            {
                if (actuals[i].Item1 != expecteds[i].Item1)
                {
                    //x coordinates must match exactly
                    return false;
                }

                if (!CloseEnough(actuals[i].Item2, expecteds[i].Item2, tolerance))
                {
                    //y values must match up to a tolerance
                    return false;
                }
            }

            return true;
        }

        private void RunGenerate(string stateFilename, string expectedFilename, double tolerance)
        {
            var outputPathname = Path.GetTempFileName();
            File.Delete(outputPathname);

            // Current folder is the bin folder, which holds the states subfolder
            RunDPlus((app, wnd) =>
            {
                OpenStateFile(app, wnd, GetPathname(stateFilename));
                DoGenerate(wnd);

                SaveOutput(app, wnd, outputPathname);
            });
            
            //Files are saved with .out extension
            outputPathname += ".out";
            var identical = IdenticalFiles(outputPathname, GetPathname(expectedFilename), tolerance);

            //Clean up the temporary file
           File.Delete(outputPathname);

            Assert.IsTrue(identical);
        }
       
        private string GetPathname(string filename)
        {
            return Path.Combine(Directory.GetCurrentDirectory(), "states", filename);
        }

        [TestMethod]
        public void TwoPops()
        {
            RunGenerate("Two Pops.state", "Two Pops_Small.out", tolerance);
        }

        [TestMethod]
        public void UHC()
        {
            RunGenerate("UHC.state", "UHC_Small.out", tolerance);
        }

        [TestMethod]
        public void PDB()
        {
            RunGenerate("PDB.state", "PDB_Small.out", tolerance);
        }
    }
}
