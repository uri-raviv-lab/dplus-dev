using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TestStack.White;
using TestStack.White.UIItems.WindowItems;

namespace TestDPlus
{
    public abstract class DPlusCaller
    {
        public TestContext TestContext { get; set; }
        public string DPlusLocation
        {
            get
            {
                return @"..\..\..\x64\release\Dplus.exe";
            }
        }

        private void AssertNoGPF(Application app)
        {
            var windows = app.GetWindows();
            Assert.AreEqual(1, windows.Where(wnd => wnd.Title == "D+").Count());
        }

        public void RunDPlus(Action<Application, Window> action)
        {
            var app = Application.Launch(DPlusLocation);
            Assert.IsNotNull(app);

            var wnd = app.GetWindow("D+");
            Assert.IsNotNull(wnd);

            try
            {
                if (wnd != null)
                    action(app, wnd);
                AssertNoGPF(app);
            }
            finally
            {
                app.Close();
            }
        }
    }
}
