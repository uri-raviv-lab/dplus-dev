#include "SPMainWindow.h"

namespace SuggestParameters
{

	System::Void SPMainWindow::textBox_Validating(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e)
	{
		TextBox ^tb = (TextBox ^)sender;

		String ^txt = tb->Text;

		Double x, y, z, q;
		bool validInput = true;
		validInput &= Double::TryParse(textBoxX->Text, x);
		validInput &= Double::TryParse(textBoxY->Text, y);
		validInput &= Double::TryParse(textBoxZ->Text, z);
		validInput &= Double::TryParse(textBoxQ->Text, q);

		if (validInput)
		{
			double maxLen = Math::Sqrt(x*x + y*y + z*z);

			double density = maxLen / Math::PI;

			int defSize = int(2 * q * density + 3);

			defSize /= 10;
			defSize++;
			defSize *= 10;

			Double largest, smallest;
			smallest = largest = x;
			largest = largest < y ? y : largest;
			largest = largest < z ? z : largest;
			smallest = smallest > y ? y : smallest;
			smallest = smallest > z ? z : smallest;

			should_be_adaptive = (largest > 5. * smallest);

			textBoxGridSize->Text = Double(defSize).ToString();

			/*
			actualGridSize = gridSize / 2 + Extras;

			long long i = actualGridSize;
			totalsz = (phiDivisions * i * (i + 1) * (3 + thetaDivisions + 2 * thetaDivisions * i)) / 6;
			totalsz++;	// Add the origin
			totalsz *= 2;	// Complex
			*/

			long long i = (defSize / 2) + 3;
			long long totalSize = (6 * i * (i + 1) * (3 + 3 + 2 * 6 * i)) / 6;
			totalSize++;
			totalSize *= 2;

			long long numBytes = sizeof(double) * totalSize;

			double mbs = double(numBytes) / (1024.*1024.);

			textBoxMemReq->Text = Int32(mbs+0.5).ToString();

			textBoxMemReq->BackColor = System::Drawing::Color::LimeGreen;
			labelWarning->Text = "";

			if (mbs > 250.)
			{
				textBoxMemReq->BackColor = System::Drawing::Color::Yellow;
				labelWarning->Text = "Note: You may want to consider using the hybrid method.";
			}

			if (mbs > 1000.)
			{
				textBoxMemReq->BackColor = System::Drawing::Color::Red;
				labelWarning->Text = "Caution: You should consider using the hybrid method.";
			}

		}

		if (Double::TryParse(textBoxQ->Text, q))
		{
			String ^subst = labelGenPoints->Text->Substring(labelGenPoints->Text->LastIndexOf(":"));
			String ^replc = ": " + Int32(q * 100).ToString();
			labelGenPoints->Text = labelGenPoints->Text->Replace(subst, replc);
		}

		checkBoxGPU_CheckedChanged(nullptr, nullptr);

	}

	System::Void SPMainWindow::checkBoxRemote_CheckedChanged(System::Object^ sender, System::EventArgs^ e)
	{
		String ^subst = labelUpdate->Text->Substring(labelUpdate->Text->LastIndexOf(":"));
		String ^replc;
		if (checkBoxRemote->Checked)
		{
			replc = ": " + Int32(1000).ToString() + "ms";
		}
		else
			replc = ": " + Int32(500).ToString() + "ms";

		
		labelUpdate->Text = labelUpdate->Text->Replace(subst, replc);

	}

	System::Void SPMainWindow::checkBoxGPU_CheckedChanged(System::Object^ sender, System::EventArgs^ e)
	{
		String ^subst = labelIntegrationMethod->Text->Substring(labelIntegrationMethod->Text->LastIndexOf(":"));
		String ^replc;
		if (checkBoxGPU->Checked)
		{
			replc = ": Adaptive (VEGAS)";
		}
		else
		{
			if (should_be_adaptive)
				replc = ": Gauss Kronrod";
			else
				replc = ": Monte Carlo";
		}
		
		labelIntegrationMethod->Text = labelIntegrationMethod->Text->Replace(subst, replc);
	}

	System::Void SPMainWindow::textBox_TextChanged(System::Object^ sender, System::EventArgs^ e)
	{
		textBox_Validating(sender, nullptr);
	}

}