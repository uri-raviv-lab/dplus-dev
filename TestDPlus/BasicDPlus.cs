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

namespace TestDPlus
{
    [TestClass]
    public class BasicDPlus: DPlusCaller
    {
        [TestMethod]
        public void CreateSphere()
        {
            RunDPlus((app, wnd) =>
            {
                var entities = wnd.Get<ComboBox>("entityCombo");
                Assert.IsNotNull(entities);
                entities.Select("Sphere");
                Assert.IsNotNull(entities.SelectedItem);

                var add = wnd.Get<Button>("buttonAdd");
                Assert.IsNotNull(add);
                add.Click();
            });
        }

        [TestMethod]
        public void CreateSymmetry()
        {
            RunDPlus((app, wnd) =>
            {
                var entities = wnd.Get<ComboBox>("entityCombo");
                Assert.IsNotNull(entities);
                entities.Select("Space-filling Symmetry");
                Assert.IsNotNull(entities.SelectedItem);

                var add = wnd.Get<Button>("buttonAdd");
                Assert.IsNotNull(add);
                add.Click();
            });
        }

        [TestMethod]
        public void CreateManualSymmetry()
        {
            RunDPlus((app, wnd) =>
            {
                var entities = wnd.Get<ComboBox>("entityCombo");
                Assert.IsNotNull(entities);
                entities.Select("Manual Symmetry");
                Assert.IsNotNull(entities.SelectedItem);

                var add = wnd.Get<Button>("buttonAdd");
                Assert.IsNotNull(add);
                add.Click();
            });
        }
    }
}
