import json

from dplus.Amplitudes import Amplitude

__author__ = "Devora Witty"
'''
Represents a state file and the state's DomainPreferences and FittingPreferences. 
Domain is located in DataModels.py
'''

import warnings
from dplus.Signal import Signal
from dplus.DataModels import Domain, Parameter

RESOLUTION_SIGMA_DEFAULT = 0.02


class DomainPreferences:
    """
    The DomainPreferences class contains properties that are copied from the D+ interface. Their usage is explained in the D+ documentation.
    """

    def __init__(self):
        self.__signal_file = ""
        self.__convergence = 0.001
        self.__grid_size = 200
        self.__orientation_iterations = 100
        self.__orientation_method = "Monte Carlo (Mersenne Twister)"
        self.__use_grid = True
        self.__apply_resolution = False
        self.__resolution_sigma = RESOLUTION_SIGMA_DEFAULT
        self.signal = Signal.create_x_vector(7.5, 0, 800)

    @property
    def q_max(self):
        '''
         The maximal value in the q range. If SignalFile is not blank,
         qMax must equal the largest x-value in the SignalFile
        '''
        return self.signal.q_max

    @q_max.setter
    def q_max(self, qmax):
        if self.signal_file:
            warnings.warn(
                "q_max must be equal to highest x value in signal file. q_max's value has not been changed")
        else:
            self.signal = Signal.create_x_vector(qmax, self.q_min, self.generated_points)

    @property
    def q_min(self):
        '''
         The maximal value in the q range. If SignalFile is not blank,
         qMax must equal the largest x-value in the SignalFile
        '''
        return self.signal.q_min

    @q_min.setter
    def q_min(self, qmin):
        if self.signal_file:
            warnings.warn(
                "q_min must be equal to lowest x value in signal file. q_min's value has not been changed")
        else:
            self.signal = Signal.create_x_vector(self.q_max, qmin, self.generated_points)

    @property
    def generated_points(self):
        '''
        The number of point to generate between qmin to qmax
         default value is 800
        '''
        return self.signal.generated_points

    @generated_points.setter
    def generated_points(self, gen):
        if self.signal_file:
            warnings.warn(
                "generated points is equal to the amount of x points in signal file. generated_points' value has not been changed")
        else:
            self.signal = Signal.create_x_vector(self.q_max, self.q_min, gen)

    @property
    def signal_file(self):
        '''
        The path to the signal file. Must be present when fitting,
            optional when generating (Generate uses the x-values from
            this file)
        '''
        return self.__signal_file

    @signal_file.setter
    def signal_file(self, sigfile):
        if not sigfile or sigfile == "":
            self.__signal_file = ""
            return
        try:
            self.signal = Signal.read_from_file(sigfile)
            self.signal = self.signal.get_validated()  # remove negative intensity values
            self.__signal_file = sigfile
        except FileNotFoundError:
            raise ValueError("signalfile must be valid file (or blank, for generate")

    @property
    def x(self):
        '''
        The number of point to generate between qmin to qmax
         default value is 800
        '''
        return self.signal.x

    @x.setter
    def x(self, x):
        self.signal = Signal(x, self.y)

    @property
    def y(self):
        '''
        The number of point to generate between qmin to qmax
         default value is 800
        '''
        return self.signal.y

    @y.setter
    def y(self, y):
        self.signal = Signal(self.x, y)

    @property
    def convergence(self):
        '''
        convergence: A convergence criteria for the orientation average. Usually
                    	10e-3 or so. See manual, chapter 6.
        '''
        return self.__convergence

    @convergence.setter
    def convergence(self, convergence):
        try:
            self.__convergence = float(convergence)
        except:
            raise ValueError("convergence must be valid number")

    @property
    def grid_size(self):
        '''
        The size of the the grid to be used (or not if !UseGrid).
        Must be a positive even integer. Minimum size is 20.
        '''
        return self.__grid_size

    @grid_size.setter
    def grid_size(self, grid_size):
        try:
            self.__grid_size = int(grid_size)
        except:
            raise ValueError("Grid size must be an integer")
        if grid_size % 2 != 0:
            raise ValueError("Grid size must be even")
        if grid_size < 20:
            raise ValueError("Minimum grid size is 20")

    @property
    def orientation_method(self):
        '''
            ["Monte Carlo (Mersenne Twister)","Adaptive (VEGAS) Monte Carlo","Adaptive Gauss Kronrod"]
            See manual chapter 6. The first is the only one implemented
            on both CPU and GPU. Selecting VEGAS makes the amplitudes
            not be saved to files (they never leave the GPU), but is
            often much faster then vanilla Monte Carlo.
        '''
        return self.__orientation_method

    @orientation_method.setter
    def orientation_method(self, method):
        if method not in ["Monte Carlo (Mersenne Twister)",
                          "Adaptive (VEGAS) Monte Carlo",
                          "Adaptive Gauss Kronrod"]:
            raise ValueError(
                "Orientation method must be either: Monte Carlo (Mersenne Twister), Adaptive (VEGAS) Monte Carlo, or Adaptive Gauss Kronrod")
        self.__orientation_method = method

    @property
    def use_grid(self):
        '''
        boolean flag for using or not using grid
        '''
        return self.__use_grid

    @use_grid.setter
    def use_grid(self, use_grid):
        if type(use_grid) == bool:
            self.__use_grid = use_grid
        else:
            raise ValueError("use_grid must be a boolean")

    @property
    def apply_resolution(self):
        '''
        boolean flag for apply or not apply resolution
        '''
        return self.__apply_resolution

    @apply_resolution.setter
    def apply_resolution(self, apply_resolution):
        if type(apply_resolution) == bool:
            self.__apply_resolution = apply_resolution
        else:
            raise ValueError("apply_resolution must be a boolean")

    @property
    def resolution_sigma(self):
        '''
        number for the parameter sigma
        '''
        return  self.__resolution_sigma

    @resolution_sigma.setter
    def resolution_sigma(self, resolution_sigma):
        if isinstance(resolution_sigma, (int, float, complex)) and not isinstance(resolution_sigma, bool) \
                and resolution_sigma >= 0:
            self.__resolution_sigma = resolution_sigma
        else:
            raise ValueError("resolution_sigma must be a positive number")

    @property
    def orientation_iterations(self):
        '''
        The number of iterations (or depth in the case of Gauss
         Krondrod) to be used in the orientation average. A good
         default value is 1,000,000. See manual, chapter 6.
        '''
        return self.__orientation_iterations

    @orientation_iterations.setter
    def orientation_iterations(self, x):
        try:
            if x <= 0:
                raise ValueError("Orientation iterations must be greater than zero")
            self.__orientation_iterations = int(x)
        except:
            raise ValueError("Orientation iterations must be integer")

    def serialize(self):
        return {
            "SignalFile": self.signal_file,
            "Convergence": self.convergence,
            "GridSize": self.grid_size,
            "UseGrid": self.use_grid,  # [true,false] Self explanatory?
            "qMin": self.q_min,
            "qMax": self.q_max,
            "OrientationIterations": self.orientation_iterations,
            "OrientationMethod": self.orientation_method,
            "ApplyResolution": self.apply_resolution,
            "ResolutionSigma": self.resolution_sigma,

            ##### irrelevant/defunct parameters:####
            "DrawDistance": 200,  # GUI parameter. Determines how far camera can "see." It's a
            # cutoff for rendering.
            "LevelOfDetail": 1,  # GUI parameter. Determines the level of detail when rendering.
            "Fitting_UpdateDomain": False,  # Deprecated and disabled in the GUI. Default should be false?
            "Fitting_UpdateGraph": True,  # Deprecated and disabled in the GUI. Default should be true to match GUI
            "UpdateInterval": 100,  # The number of milliseconds that the frontend waits before
            "GeneratedPoints": self.generated_points,  # The length of x vector
            #	polling the backend for progress. Broken.
        }

    def load_from_dictionary(self, in_dict):
        """
        sets the values of the various fields within a class to match those contained within a suitable dictionary.

        :param in_dict: json dictionary
        """
        self.signal_file = in_dict.get("SignalFile", "")
        self.convergence = in_dict["Convergence"]
        self.grid_size = in_dict["GridSize"]
        self.use_grid = in_dict["UseGrid"]
        self.resolution_sigma = in_dict.get("ResolutionSigma", RESOLUTION_SIGMA_DEFAULT)
        self.apply_resolution = in_dict.get("ApplyResolution", False)

        if not self.signal_file:
            q_min = in_dict.get("qMin", 0)
            q_max = in_dict["qMax"]
            generated_points = in_dict.get("GeneratedPoints", 800)
            self.signal = Signal.create_x_vector(q_max, q_min, generated_points)

        self.orientation_iterations = in_dict["OrientationIterations"]
        self.orientation_method = in_dict["OrientationMethod"]

    def __str__(self):
        return (str(self.serialize()))


class FittingPreferences:
    def __init__(self):
        self.__Convergence = 0.1
        self.__DerEps = 0.1
        self.__FittingIterations = 20
        self.__LossFuncPar1 = 0.5
        self.__LossFuncPar2 = 0.5
        self.__LossFunction = "Trivial Loss"
        self.__StepSize = 0.01
        self.__XRayResidualsType = "Normal Residuals"
        self.__DoglegType = "Traditional Dogleg"
        self.__LineSearchDirectionType = "L-BFGS"
        self.__LineSearchType = "Wolfe"
        self.__MinimizerType = "Trust Region"
        self.__NonlinearConjugateGradientType = "Fletcher Reeves"
        self.__TrustRegionStrategyType = "Levenberg-Marquardt"

    @property
    def loss_function(self):
        '''
        See http://ceres-solver.org/nnls_tutorial.html for explanation.
        acceptable values: "Trivial Loss","Huber Loss","Soft L One Loss","Cauchy Loss","Arctan Loss","Tolerant Loss"
        '''
        return self.__LossFunction

    @loss_function.setter
    def loss_function(self, loss_func):
        '''
        :param loss_func: acceptable values: "Trivial Loss","Huber Loss","Soft L One Loss","Cauchy Loss","Arctan Loss","Tolerant Loss"
           all loss functions except Trivial require one parameter (default 0.5)
        Tolerant loss requires a second parameter (will be ignored for all other types) (default 0.5)
        '''
        if loss_func not in ["Trivial Loss", "Huber Loss", "Soft L One Loss",
                             "Cauchy Loss", "Arctan Loss", "Tolerant Loss"]:
            raise ValueError("Loss function not valid. Please choose from:"
                             "Trivial, Huber, Soft L One, Cauchy, Arctan, or Tolerant Loss")

        self.__LossFunction = loss_func

    @property
    def loss_func_param_one(self):
        '''
        required for all loss functions but Trivial
        '''
        return self.__LossFuncPar1

    @loss_func_param_one.setter
    def loss_func_param_one(self, x):
        self.__LossFuncPar1 = x

    @property
    def loss_func_param_two(self):
        '''
        required for Tolerant Loss function
        '''
        return self.__LossFuncPar2

    @loss_func_param_two.setter
    def loss_func_param_two(self, x):
        self.__LossFuncPar2 = x

    @property
    def convergence(self):
        '''
        # The convergence criteria passed on to Ceres that determines
        #	when the fitting process has converged.
        #	Required.
        '''
        return self.__Convergence

    @convergence.setter
    def convergence(self, x):
        if x > 0:
            self.__Convergence = x
        else:
            raise ValueError("convergence must be positive")

    @property
    def x_ray_residuals_type(self):
        '''
        # ["Normal Residuals","Ratio Residuals","Log Residuals"]
            #	A function that determines how the residuals are treated.
            #	Apparently not documented anywhere. Corresponds to [XRayResiduals,
            #	XRayRatioResiduals,XRayLogResiduals] in modelfitting.cpp.
            #	Required.
        '''
        return self.__XRayResidualsType

    @x_ray_residuals_type.setter
    def x_ray_residuals_type(self, xtype):
        if xtype not in ["Normal Residuals", "Ratio Residuals", "Log Residuals"]:
            raise ValueError("x ray residual type must be Normal, Ratio, or Lod Residuals")
        self.__XRayResidualsType = xtype

    @property
    def fitting_iterations(self):
        '''
  # The number of iterations the fitter should use before stopping.
            #	Note that this is also the number of "inner iterations" that
            #	ceres uses (http://ceres-solver.org/nnls_solving.html#inner-iterations).
        '''
        return self.__FittingIterations

    @fitting_iterations.setter
    def fitting_iterations(self, x):
        if x > 0 and float(x).is_integer():
            self.__FittingIterations = x
        else:
            raise ValueError("Fitting iterations must be positive integer")

    @property
    def step_size(self):
        '''
        # Parameter for ceres. Not entirely sure how this is used,
            #	beyond that it's used for computing derivatives.
        '''
        return self.__StepSize

    @step_size.setter
    def step_size(self, x):
        self.__StepSize = x

    @property
    def der_eps(self):
        '''
        # Parameter for ceres. Not entirely sure how this is used,
            #	beyond that it's used for computing derivatives.
        '''
        return self.__DerEps

    @der_eps.setter
    def der_eps(self, x):
        self.__DerEps = x

    @property
    def minimizer_type(self):
        '''
        Required.
        Valid values: ["Line Search","Trust Region"]
            #	If "Trust Region", TrustRegionStrategyType must be valid.
            #	If "Line Search", lineSearchType must be valid.
        '''
        return self.__MinimizerType

    @minimizer_type.setter
    def minimizer_type(self, min):
        if min not in ["Line Search", "Trust Region"]:
            raise ValueError("minimizer type must be Line Search or Trust Region")
        self.__MinimizerType = min

    @property
    def trust_region_strategy_type(self):
        '''
            # ["Levenberg-Marquardt","Dogleg"]
            #	If "Dogleg", then DoglegType must be valid.
        '''
        return self.__TrustRegionStrategyType

    @trust_region_strategy_type.setter
    def trust_region_strategy_type(self, strat):
        if strat != "":
            if strat not in ["Levenberg-Marquardt", "Dogleg"]:
                raise ValueError("Trust region strategy must be either Levenberg-Marquadt or Dogleg")
        self.__TrustRegionStrategyType = strat

    @property
    def dogleg_type(self):
        '''
        valid values: # ["Traditional Dogleg","Subspace Dogleg"]
        '''
        return self.__DoglegType

    @dogleg_type.setter
    def dogleg_type(self, dog):
        if dog != "":
            if dog not in ["Traditional Dogleg", "Subspace Dogleg"]:
                raise ValueError("dogleg type must be Traditional or Subspace Dogleg")
        self.__DoglegType = dog

    @property
    def line_search_type(self):
        # ["Armijo","Wolfe"]
        #	If "Armijo", then LineSearchDirectionType cannot be ["BFGS","L-BFGS"].
        return self.__LineSearchType

    @line_search_type.setter
    def line_search_type(self, line):
        if line != "":
            if line not in ["Armijo", "Wolfe"]:
                raise ValueError("line search type must be Armijo or Wolfe")
        self.__LineSearchType = line

    @property
    def line_search_direction_type(self):
        '''
        # ["Steepest Descent","Nonlinear Conjugate Gradient","L-BFGS","BFGS"]
        #	If "Nonlinear Conjugate Gradient", NonlinearConjugateGradientType
        #	must be valid.
        '''
        return self.__LineSearchDirectionType

    @line_search_direction_type.setter
    def line_search_direction_type(self, dir):
        if dir != "":
            if dir not in ["Steepest Descent", "Nonlinear Conjugate Gradient", "L-BFGS", "BFGS"]:
                raise ValueError("line search direction type must be "
                                 "Steepest Descent, Nonlinear Conjugate Gradient,L-BFGS, or BFGS")
        self.__LineSearchDirectionType = dir

    @property
    def nonlinear_conjugate_gradient_type(self):
        '''
            # ["Fletcher Reeves","Polak Ribirere","Hestenes Stiefel"]
            #	See ceres documentation.
        '''
        return self.__NonlinearConjugateGradientType

    @nonlinear_conjugate_gradient_type.setter
    def nonlinear_conjugate_gradient_type(self, grad):
        if grad != "":
            if grad not in ["Fletcher Reeves", "Polak Ribirere", "Hestenes Stiefel"]:
                raise ValueError("gradient type must be Fletcher Reeves, Polak Ribirere, or Hestenes Stiefel")
        self.__NonlinearConjugateGradientType = grad

    def serialize(self):
        self.validate()  # before returning dictionary to be used in further calculation,
        # check that values in dictionary are valid.
        return {

            "Convergence": self.convergence,
            "XRayResidualsType": self.x_ray_residuals_type,
            "FittingIterations": self.fitting_iterations,
            "StepSize": self.step_size,
            "DerEps": self.der_eps,

            "LossFunction": self.loss_function,
            "LossFuncPar1": self.loss_func_param_one,
            "LossFuncPar2": self.loss_func_param_two,

            "MinimizerType": self.minimizer_type,

            "TrustRegionStrategyType": self.trust_region_strategy_type,

            "DoglegType": self.dogleg_type,

            "LineSearchType": self.line_search_type,
            "LineSearchDirectionType": self.line_search_direction_type,
            "NonlinearConjugateGradientType": self.nonlinear_conjugate_gradient_type,

        }

    def validate(self):
        if self.minimizer_type == "Trust Region":
            if self.trust_region_strategy_type not in ["Levenberg-Marquardt", "Dogleg"]:
                raise ValueError("If minimizer type is trust region, must set trust_region_strategy_type")
            if self.trust_region_strategy_type == "Dogleg":
                if self.dogleg_type not in ["Traditional Dogleg", "Subspace Dogleg"]:
                    raise ValueError("If trust region strategy type is dogleg, must set dogleg_type")

        if self.minimizer_type == "Line Search":
            if self.line_search_type not in ["Armijo", "Wolfe"]:
                raise ValueError("If trust region type is line search, must set line_search_type")
            if self.line_search_type == "Armijo":
                if self.line_search_direction_type in ["BFGS", "L-BFGS"]:
                    raise ValueError("If line search type is Armijo, line search direction type cannot be BFGS, L-BFGS")
            if self.line_search_direction_type == "Nonlinear Conjugate Gradient":
                if self.nonlinear_conjugate_gradient_type not in ["Fletcher Reeves", "Polak Ribirere",
                                                                  "Hestenes Stiefel"]:
                    raise ValueError(
                        "if line search direction type is nonlinear conjugate gradient, must set nonlinear_conjugate_gradient_type")

    def load_from_dictionary(self, json):
        self.convergence = json["Convergence"]
        self.x_ray_residuals_type = json["XRayResidualsType"]
        self.fitting_iterations = json["FittingIterations"]
        self.step_size = json["StepSize"]
        self.der_eps = json["DerEps"]

        self.loss_function = json["LossFunction"]
        self.loss_func_param_one = json["LossFuncPar1"]
        self.loss_func_param_two = json["LossFuncPar2"]

        self.minimizer_type = json["MinimizerType"]

        self.trust_region_strategy_type = json["TrustRegionStrategyType"]
        self.dogleg_type = json["DoglegType"]

        self.line_search_type = json["LineSearchType"]
        self.line_search_direction_type = json["LineSearchDirectionType"]
        self.nonlinear_conjugate_gradient_type = json["NonlinearConjugateGradientType"]
        self.validate()

    def __str__(self):
        return (str(self.serialize()))


class State:
    """
    The state class contains an instance of each of three classes: DomainPreferences, FittingPreferences, and Domain.
    """

    def __init__(self):
        self.DomainPreferences = DomainPreferences()
        self.FittingPreferences = FittingPreferences()
        self.__Viewport = {  # This is totally irrelevant and is here for legacy purposes only
            "Axes_at_origin": True,
            "Axes_in_corner": True,
            "Pitch": 35.264389038086,
            "Roll": 0,
            "Yaw": 225.00001525879,
            "Zoom": 8.6602535247803,
            "cPitch": 35.264389038086,
            "cRoll": 0,
            "cpx": -4.9999990463257,
            "cpy": -5.0000004768372,
            "cpz": 4.9999995231628,
            "ctx": 0,
            "cty": 0,
            "ctz": 0
        }
        self.Domain = Domain()
        self.__filenames = []

    def load_from_dictionary(self, json):
        """
        sets the values of the various fields within a class to match those contained within a suitable dictionary. \
        It can behave recursively as necessary, for example with a model that has children.

        :param json: json dictionary
        """
        self.DomainPreferences.load_from_dictionary(json["DomainPreferences"])
        self.FittingPreferences.load_from_dictionary(json["FittingPreferences"])
        try:
            self.__Viewport = json["Viewport"]
        except:
            pass  # no one cares about viewport
        self.Domain = Domain()
        self.Domain.load_from_dictionary(json["Domain"])

    def serialize(self):
        """
        saves the contents of a class to a dictionary.

        :return: dictionary of the class fields (DomainPreferences, FittingPreferences and Domain)
        """
        self._validate_model_tree()
        return {
            "DomainPreferences": self.DomainPreferences.serialize(),
            "FittingPreferences": self.FittingPreferences.serialize(),
            "Viewport": self.__Viewport,
            "Domain": self.Domain.serialize()
        }

    def export_all_parameters(self, filename):
        """
           call the function serialize and save the result (dictionary of the class fields) to a given filename

          :param filename: filename as string
          """
        with open(filename, 'w') as file:
            file.write(json.dumps(self.serialize()))

    def __str__(self):
        return (
                "Domain Preferences: " + str(self.DomainPreferences)
                + "\n"
                + "Fitting Preferences: " + str(self.FittingPreferences)
                + "\n"
                + "Domain: " + str(self.Domain)
        )

    def get_model(self, name_or_ptr):
        """
          return a model from the Domain field by either its `name` or its `model_ptr`.

          :rtype: an instance of Model
          """

        if not isinstance(name_or_ptr, int) and not isinstance(name_or_ptr, str):
            raise ValueError("Either provide an integer model pointer or a string model name")

        num_found = 0
        if self.Domain.model_ptr == name_or_ptr:
            result = self.Domain
            num_found += 1
        for population in self.Domain.populations:
            if population.model_ptr == name_or_ptr:
                result = population
                num_found += 1
            for model in population.models:
                inner_result = self.get_model_recursive(model, name_or_ptr)
                if inner_result is not None:
                    result = inner_result
                    num_found += 1
        if num_found == 0:
            raise (Exception("Model with that identifier not found"))
        if num_found > 1:
            raise (Exception("Model identifier not unique. More than one model found"))
        return result

    def get_model_recursive(self, model, name_or_ptr):
        if hasattr(model, 'name') and hasattr(model, 'model_ptr'):
            if model.name == name_or_ptr or model.model_ptr == name_or_ptr:
                return model
        if hasattr(model, 'Name') and hasattr(model, 'ModelPtr'):
            if model.Name == name_or_ptr or model.ModelPtr == name_or_ptr:
                return model
        if not hasattr(model, 'Children') and not hasattr(model, 'children'):
            return None
        if len(model.children) == 0:
            return None
        for child in model.children:
            result = self.get_model_recursive(child, name_or_ptr)
            if result is not None:
                return result
        return None

    def validate_use_grid(self):
        """
        The function check all the "use_grid" in the tree, and changes them to False if it's needed.
        If one model has use_grid==False, all the model's parents should be False too.
        :return:
        """
        [_, model_dict] = self.validate_use_grid_recursive(self.Domain, {})
        if model_dict:
            raise ValueError("There are models that have use_grid==True,"
                             " but they have a child with use_grid==False. model_ptr:{},"
                             "please change them to False or fix their child to True ".format(model_dict.keys()))

    def _fix_use_grid(self):
        """
        The function check all the "use_grid" in the tree, and changes them to False if it's needed.
        If one model has use_grid==False, all the model's parents should be False too.
        :return:
        """
        [_, model_dict] = self.validate_use_grid_recursive(self.Domain, {})
        for model_ptr, model in model_dict.items():
            model.use_grid = False

    def validate_use_grid_recursive(self, model, dict_res):
        """
        If one model has use_grid==False, all the model's parents should be False too.
        The function return the dict of the models that need to be use_grid==False
        :param model: The model we are testing if needs to change
        :param dict_res: dictionary for represent the models that needed change
        :return: add_parent: bool value if the model has child with "false", dict_res: the dictionary of the models
        """
        add_parent = False
        if not hasattr(model, 'children') or len(model.children) == 0:
            return not model.use_grid, {}
        for child in model.children:
            add_parent, dict_tmp = self.validate_use_grid_recursive(child, dict_res)
            dict_res = {**dict_tmp, **dict_res}
            if add_parent:
                if hasattr(model, 'use_grid') and model.use_grid:
                    dict_res[model.model_ptr] = model
        return add_parent, dict_res

    def get_models_by_type(self, type):
        """
          returns a list of `Models` from the Domain field with a given `type_name`.

          :param type: a string of model type , e.g. UniformHollowCylinder.
          :rtype: list of instances of 'Model'
          """

        models = []
        for population in self.Domain.children:
            for model in population.children:
                self.get_model_by_type_recursive(model, type, models)
                # if model.type_name == type:
                #     models.append(model)
        return models

    def get_model_by_type_recursive(self, model, type, models_list):
        if hasattr(model, '_metadata') and model._metadata["type_name"] == type:
            models_list.append(model)
        if not hasattr(model, 'children'):
            return
        if len(model.children) == 0:
            return
        for child in model.children:
            self.get_model_by_type_recursive(child, type, models_list)
        return

    def get_mutable_parameter_values(self):
        """
          returns the values of all the mutable parameters in the Domain field.

          :rtype: a list of floats
          """

        params = self.get_mutable_params()
        param_array = []
        for model_params in params:
            for param in model_params:
                param_array.append(float(param.value)) # convert to float for ignore case is int32, that make problems later
        return param_array

    def get_mutable_parameter_options(self):
        params = self.get_mutable_params()
        sigma_array = []
        constr_min = []
        constr_max = []
        for model_params in params:
            for param in model_params:
                sigma_array.append(param.sigma)
                constr_min.append(param.constraints.min_value)
                constr_max.append(param.constraints.max_value)
        bounds= tuple([constr_min, constr_max])
        return sigma_array, bounds

    def set_mutable_parameter_values(self, param_vals_array):
        """
           sets the mutable parameters values in the Domain field with the the input list (in the order given by \
           get_mutable_parameter_values).

           :param param_vals_array: a list of floats
           """


        def _set_recursive(param_array, model):
            mut_len = len(model.get_mutable_params())
            model.set_mutable_params(param_array[:mut_len])
            try:
                used_values = 0
                for child in model.children:
                    used_values += _set_recursive(param_array[mut_len + used_values:], child)
            except AttributeError as E:
                pass
            return mut_len + used_values

        _set_recursive(param_vals_array, self.Domain)


    def get_mutable_params(self):
        '''
        return array of arrays of all mutable parameters in tree.

        :rtype: a list of `Parameters`
        '''

        def _recursive(param_array, model):
            mut = model.get_mutable_params()
            if mut:
                param_array.append(mut)
            try:
                for child in model.children:
                    models_set = _recursive(param_array, child)
            except AttributeError:
                pass
            return param_array

        res = _recursive([], self.Domain)
        return [par for par in res if par is not None]

    def get_mutable_params_array(self):
        '''
        return array of arrays of all mutable parameters in array.

        :rtype: a list of `Parameters`
        '''

        def _recursive(param_array, model):
            mut = model.get_mutable_params()
            if mut:
                param_array.extend(mut)
            try:
                for child in model.children:
                    models_set = _recursive(param_array, child)
            except AttributeError:
                pass
            return param_array

        res = _recursive([], self.Domain)
        return [par for par in res if par is not None]

    def _validate_all_models_indices(self):
        def _recursive(models_set, model):
            if model.model_ptr in models_set:
                raise ValueError("There are non-unique model pointers")
            models_set.add(model.model_ptr)
            try:
                for child in model.children:
                    models_set = _recursive(models_set, child)
            except AttributeError:
                pass
            return models_set

        return _recursive(set(), self.Domain)

    def _get_all_filenames(self):
        # TODO: This is very not up to date with new file name passing
        def _recursive(model):
            try:
                self.__filenames += [filename for filename in model.filenames if filename not in ("", None)]
            except AttributeError:
                pass
            try:
                for child in model.children:
                    _recursive(child)
            except AttributeError:
                pass

        self.__filenames = []
        _recursive(self.Domain)
        if self.DomainPreferences.signal_file not in ["", None]:
            self.__filenames += [self.DomainPreferences.signal_file]
        return self.__filenames

    def _validate_model_tree(self):
        # TODO: add other validations beyond model ptr unqiueness
        '''
        validations that need to be added:
        layers between min and max allowed amount
        children only on models with children, parameters only on models with parameters
        only parameters that exist in the model allowed
        !!model_ptr must be unique!!

        '''

        man_symms = self.get_models_by_type("Manual Symmetry")
        for model in man_symms:
            if not model.layer_params:
                raise ValueError("Manual Symmetry model must include at least one layer. model_ptr:{}".format(model.model_ptr))
        self._validate_all_models_indices()

    def add_model(self, model, population_index=0):
        """
        is a convenience function to help add models to the state's parameter tree. It receives the model and optionally \
        a population index (default 0), and will insert that model into the population.

        :param model: instance of 'Model' type
        :param population_index: int value
        """
        self.Domain.populations[population_index].children.append(model)

    def add_amplitude(self, amplitude, population_index=0):
        """
        is a convenience function specifically for adding instances of the `Amplitude` class. \
        It creates an instance of `AMP` with the `Amplitude`'s filename, and then in addition to calling `add_model` with that AMP, \
        it also changes the state's DomainPreferences (specifically grid_size, q_max, and use_grid) to match the Amplitude's properties.\
        It returns the AMP it created.

        :param amplitude:  instance of `Amplitude`
        :param population_index: int value that indicate where to add the new model that was created from the amplitude
        :return: `AMP` type model
        """
        from dplus.DataModels.models import AMP
        if not isinstance(amplitude, Amplitude):
            raise ValueError(
                "add_amplitude can only be called on Amplitudes created from the dplus.Amplitudes.Amplitude class")
        if not amplitude.filename:
            raise ValueError("you must save the Amplitude to a file before adding it to the parameter tree")

        A = AMP(amplitude.filename)

        if not self.DomainPreferences.use_grid:
            self.DomainPreferences.use_grid = True
            print("Set Domain Preferences use_grid to True")

        if self.DomainPreferences.grid_size != amplitude.grid.grid_size:
            self.DomainPreferences.grid_size = amplitude.grid.grid_size
            print("Set DomainPerefences grid_size to " + str(amplitude.grid.grid_size))

        if self.DomainPreferences.q_max != amplitude.grid.q_max:
            self.DomainPreferences.q_max = amplitude.grid.q_max
            print("Set DomainPreferences q_max to " + str(amplitude.grid.q_max))

        self.add_model(A, population_index)
        return A

    def _get_dplus_fit_results_json(self):
        '''
        D+ expects the results of fitting to be returned in a special json format (used nowhere else)
        that the D+ code refers to as "simple"

        Not part of the Python Interface public API, as it's only use is when integrating with D+
        '''
        orientation_dict = {
            "Monte Carlo (Mersenne Twister)": 0,
            "Adaptive (VEGAS) Monte Carlo": 1,
            "Adaptive Gauss Kronrod": 2,
            "Direct Computation - MC": 3,
            "Monte Carlo (Sobol) - unimplemented": 4
        }

        basic_dict = self.Domain._basic_json_params(self.DomainPreferences.use_grid)

        # the populations have domain preferences added as parameters (i have no idea why),
        # and hence their dictionary needs to be modified here in the state
        # this is not necessarily correct for singleGeometry, I'm not sure how to handle that
        for population_dict in basic_dict["Submodels"]:
            population_dict["Parameters"].append(Parameter(self.DomainPreferences.orientation_iterations, name="orientation iterations").serialize())
            population_dict["Parameters"].append(Parameter(self.DomainPreferences.grid_size, name="grid_size").serialize())
            use_grid_val= 1 if self.DomainPreferences.use_grid else 0
            population_dict["Parameters"].append(Parameter(use_grid_val, name="use_grid").serialize())
            population_dict["Parameters"].append(Parameter(self.DomainPreferences.convergence, name="covergence").serialize())
            population_dict["Parameters"].append(Parameter(self.DomainPreferences.q_max, name="q_max").serialize())
            population_dict["Parameters"].append(Parameter(orientation_dict[self.DomainPreferences.orientation_method], name="orientation method").serialize())  ##expects to receive an ENUM translation
            population_dict["Parameters"].append(Parameter(self.DomainPreferences.q_min, name="q_min").serialize())
        return basic_dict