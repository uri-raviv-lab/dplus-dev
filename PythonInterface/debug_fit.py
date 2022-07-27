# Try to debug the fitting process
# Target: Get the Python fit to work exactly like the D+ fit

from dplus.FitRunner import FitRunner
from dplus.CalculationInput import CalculationInput

STATE_FILE = "d:\\temp\\dplus\\Sphere_Radius_and_ED_Fit.state"
SIGNAL_FILE = "d:\\temp\\dplus\\1Sphere_GK_10.dat"

def main():
    input = CalculationInput.load_from_state_file(STATE_FILE)
    # input.FittingPreferences.fitting_iterations = 1000
    runner = FitRunner()
    results = runner.fit(input)

if __name__ == '__main__':
    main()