#include "SPMainWindow.h"

/**
 * @file SPMainWindow.cpp
 * @brief Implements the SPMainWindow class for the Suggest Parameters tool, providing UI logic for parameter validation,
 *        grid size estimation, memory usage calculation, and integration method selection.
 *
 * The SPMainWindow module is responsible for:
 *  - Validating user input for parameter fields (X, Y, Z, Q) and updating dependent UI fields accordingly.
 *  - Estimating grid size and memory requirements based on user input, and providing visual feedback and warnings.
 *  - Automatically suggesting the appropriate integration method (adaptive, Gauss Kronrod, or Monte Carlo) based on input.
 *  - Adjusting UI labels and settings in response to user actions (e.g., toggling GPU or remote computation).
 *
 * Key Concepts:
 *  - Parameter Validation: Ensures that user-entered values for X, Y, Z, and Q are valid doubles before updating calculations.
 *  - Grid Size and Memory Estimation: Calculates recommended grid size and memory requirements for computations, with color-coded warnings.
 *  - Integration Method Selection: Chooses the integration method based on input values and GPU/remote settings.
 *  - UI Feedback: Updates labels, text boxes, and warnings in real time as the user interacts with the form.
 *
 * Fields:
 *  - TextBox^ textBoxX, textBoxY, textBoxZ, textBoxQ, textBoxGridSize, textBoxMemReq:
 *      UI fields for entering and displaying parameter values and computed results.
 *  - Label^ labelWarning, labelGenPoints, labelUpdate, labelIntegrationMethod:
 *      UI labels for warnings, generated points, update interval, and integration method.
 *  - CheckBox^ checkBoxGPU, checkBoxRemote:
 *      UI checkboxes for toggling GPU and remote computation.
 *  - bool should_be_adaptive:
 *      Indicates if the adaptive integration method should be suggested based on parameter ratios.
 *
 * Main Methods:
 *  - textBox_Validating: Validates parameter input, updates grid size and memory estimates, and adjusts warnings.
 *  - checkBoxRemote_CheckedChanged: Updates the update interval label based on remote computation setting.
 *  - checkBoxGPU_CheckedChanged: Updates the integration method label based on GPU and parameter settings.
 *  - textBox_TextChanged: Triggers validation and recalculation when any parameter text box changes.
 *
 * Threading and Safety:
 *  - All UI updates are performed on the main thread; not thread-safe for background access.
 *
 * Error Handling:
 *  - Provides visual feedback (color changes, warning labels) for high memory usage or invalid input.
 *  - Ensures calculations are only performed when all required inputs are valid.
 *
 * Dependencies:
 *  - SPMainWindow.h for class and method declarations.
 *  - System::Windows::Forms for UI controls and events.
 *
 * See SPMainWindow.h for class and method declarations.
 */

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