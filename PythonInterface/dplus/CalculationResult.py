import pprint
from collections import OrderedDict
from dplus.CalculationInput import CalculationInput
from dplus.Signal import Signal


class CalculationResult(object):
    """
    Stores the various aspects of the result for further manipulation
    """

    def __init__(self, calc_data, result, job ):
        '''

        :param calc_data: an instance of CalculationInput class
        :param result: a json with the result of fit/ generate
        :param job: an instance of RunningJob
        '''
        self._raw_result = result  # a json
        self._job = job  # used for getting amps and pdbs
        self._calc_data = calc_data  # gets x for getting graph. possibly also used for fitting.
        self._headers = OrderedDict()

        if 'Graph' not in self._raw_result:
            # sometimes fit doesn't return graph, also any time generate crashes
            print("No graph returned")
        elif len(self._calc_data.x) != len(self._raw_result['Graph']):
            raise ValueError("Result graph size mismatch")
        else:
            self.signal = Signal(self._calc_data.x, self._raw_result['Graph'])


    @property
    def processed_result(self):
        res=self._raw_result
        res['Graph']=list(self.signal.y)
        return res


    def __str__(self):
        return pprint.pformat(self._raw_result)

    @property
    def graph(self):
        return self.signal.graph
    @property
    def y(self):
        '''

        :return: The raw list of intensity values from the results json
        '''
        return self.signal.y

    @property
    def headers(self):
        '''
        :return: an OrderDict of headers, whose keys are ModelPtrs and whose values are the header associated.
        '''
        return self._headers

    def get_pdb(self, model_ptr, destination_folder=None):
        '''
        returns the file location of the pdb file for given model_ptr. \
        destination_folder has a default value of None, but if provided, the pdb file will be copied to that location,\
        and then have its address returned
        :param model_ptr: int value of model_ptr
        :param destination_folder: location to copy the pdb file of the given model_ptr
        :return: File location of the pdb file
        '''
        return self._job._get_pdb(model_ptr, destination_folder)

    def get_amp(self, model_ptr, destination_folder=None):
        '''
           returns the file location of the amplitude file for given model_ptr. \
           destination_folder has a default value of None, but if provided, the amplitude file will be copied to that location,\
           and then have its address returned.

          :param model_ptr: int value of model_ptr
          :param destination_folder: location to copy the amplitude file of the given model_ptr
          :return: File location of the amplitude file
          '''
        model_name = self._calc_data.get_model(model_ptr).name
        return self._job._get_amp(model_ptr, model_name, destination_folder)

    def get_amps(self, destination_folder=None):
        '''
           fetches all the amplitude files created by the calculation, and returns an array of their folder locations. \
           destination_folder has a default value of None, but if provided, the amplitude files will be copied to that folder\

          :param destination_folder: optional location to save the amplitude files to
          :return: Array of file locations of the amplitude files
          '''

        addresses = []
        for model_ptr in self._calc_data._validate_all_models_indices():
            try:
                addresses.append(self.get_amp(model_ptr, destination_folder))
            except FileNotFoundError:  # not every model will necessarily have an amplitude file
                pass
        return addresses

    @property
    def error(self):
        '''
        :return: returns the json error report from the dplus run
        '''
        if "error" in self._raw_result:
            return self._raw_result["error"]
        return {"code": 0, "message": "no error"}

    def save_to_out_file(self, filename):
        '''
        receives file name, and saves the results to the file.
        :param filename: string of filename/path
        '''
        with open(filename, 'w') as out_file:
            domain_preferences = self._calc_data.DomainPreferences
            out_file.write("# Integration parameters:\n")
            out_file.write("#\tqmax\t{}\n".format(domain_preferences.q_max))
            out_file.write("#\tOrientation Method\t{}\n".format(domain_preferences.orientation_method))
            out_file.write("#\tOrientation Iterations\t{}\n".format(domain_preferences.orientation_iterations))
            out_file.write("#\tConvergence\t{}\n\n".format(domain_preferences.convergence))

            for value in self.headers.values():
                out_file.write(value)
            for key, value in self.graph.items():
                out_file.write('{:.5f}\t{:.20f}\n'.format(key, value))
            out_file.close()


class GenerateResult(CalculationResult):
    '''
    A class for generate calculation results
    '''

    def __init__(self, calc_data, result, job):
        super().__init__(calc_data, result, job)
        if self._calc_data.DomainPreferences.apply_resolution:
            self.signal = self.signal.apply_resolution_function(self._calc_data.DomainPreferences.resolution_sigma)
        self._parse_headers()  # sets self._headers to a list of headers

    def _parse_headers(self):
        header_dict = OrderedDict()
        try:
            headers = self._raw_result['Headers']
            for header in headers:
                header_dict[header['ModelPtr']] = header['Header']
        except:  # TODO: headers don't appear in fit results?
            pass  # regardless, I'm pretty sure no one cares about headers anyway
        self._headers = header_dict


class FitResult(CalculationResult):
    '''
    A class for fit calculation results
    '''

    def __init__(self, calc_data, result, job):
        super().__init__(calc_data, result, job)
        self._get_parameter_tree()  # right now just returns value from result.
        self.create_state_results()

    def _get_parameter_tree(self):
        try:
            self._parameter_tree = self._raw_result['ParameterTree']
        except KeyError:
            raise Exception("ParameterTree doesn't exist")

    @property
    def parameter_tree(self):
        '''
        A json of parameters (can be used to create a new state with state's load_from_dictionary).
        :return: A json of parameters
        '''
        return self._parameter_tree

    def create_state_results(self):
        '''
        This function creates CalculationInput class from the parameters tree returned from a Fit calculation
        :return:
        '''

        # Combine results returned from a Fit calculation
        def combine_model_parameters(parameters):
            # Combine parameters of just one model
            model_ptr = parameters['ModelPtr']
            model = self.__state_result.get_model(model_ptr)
            mutables = model.get_mutable_params() or []
            updated = 0
            for param in parameters['Parameters']:
                if param['isMutable']:
                    if updated >= len(mutables):
                        raise ValueError("Found more 'isMutable' params in ParameterTree than in our state")
                    mutables[updated].value = param['Value']
                    updated += 1
            if updated != len(mutables):
                raise ValueError(
                    "Found a mismatch between number of 'isMutable' params in the ParamterTree and in our state")

        def recursive(parameters):
            combine_model_parameters(parameters)
            for sub in parameters['Submodels']:
                recursive(sub)

        self.__state_result = CalculationInput.copy_from_state(self._calc_data)
        recursive(self.parameter_tree)

    @property
    def result_state(self):
        return self.__state_result
