using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Deployment.WindowsInstaller;
using System.Windows.Forms;
namespace CustomActionAVX
{
    public class CustomActions
    {
        [CustomAction]
        public static ActionResult CustomActionCheckAVX(Session session)
        {
            try
            {
                session.Log("Begin Check AVX Custom Action");
                    bool result = (GetEnabledXStateFeatures() & 4) != 0;
                    if (!result)
                    {

                        MessageBox.Show("Installation Error : You're using an old CPU, CPU requires an Intel CPU from 2011 or later", "Installation Error");
                        session.Log("ERROR in custom action CheckAVX: {0}", "You're using an old CPU, CPU requires an Intel CPU from 2011 or later");
                        return ActionResult.Failure;
                    }

                session.Log("End Check AVX Custom Action");
            }
            catch (Exception ex)
            {
                MessageBox.Show("ERROR in custom action CheckAVX " + ex.ToString(), "Installation Error");
                session.Log("ERROR in custom action CheckAVX {0}", ex.ToString());
                return ActionResult.Failure;
            }

            session.Log("custom action CheckAVX {0}", "Success");
            return ActionResult.Success;
        }
        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
        private static extern long GetEnabledXStateFeatures();
    }
}
