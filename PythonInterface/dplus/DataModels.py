import math
import sys

from dplus.metadata import meta_models, hardcode_models, _type_to_int, _models_with_files_index_dict, _int_to_type


class Constraints:
    '''
    The Constraints class contains the following properties:

    * MaxValue: a float whose default value is infinity
    * MinValue: a float whose default value is -infinity
    '''

    def __init__(self, max_val=math.inf, min_val=-math.inf, minindex=-1, maxindex=-1, link=-1):
        try:
            if max_val <= min_val:
                raise ValueError("Constraints' upper bound must be greater than lower bound")
        except TypeError:  # for some reason, strings weren't being converted to numbers properly
            if max_val == "inf":
                max_val = math.inf
            if min_val == "-inf":
                min_val = -math.inf
            if max_val <= min_val:
                raise ValueError("Constraints' upper bound must be greater than lower bound")
        self.MaxValue = max_val
        self.MinValue = min_val
        self.isConstrained = False
        if max_val != math.inf or min_val != -math.inf:
            self.isConstrained = True
        self.min_index = minindex
        self.max_index = maxindex
        self.link = link

    @classmethod
    def from_dictionary(cls, json):
        """
        creates Constraints class instance with the json dictionary.
        :param json: json dictionary
        :return instance of Constraints class with the json data
        """
        try:
            c = cls(json["MaxValue"], json["MinValue"], json["MinIndex"], json["MaxIndex"], json["Link"])
        except KeyError:  # backwards compatibility with older version of constraints
            c = cls(json["MaxValue"], json["MinValue"])
        return c

    def serialize(self):
        """
        saves the contents of a class to a dictionary.

        :return: dictionary of the class fields (isConstrained, consMin and consMax)
        """
        return {
            "Link": self.link,
            "MaxIndex": self.max_index,
            "MaxValue": self.MaxValue,
            "MinIndex": self.min_index,
            "MinValue": self.MinValue
        }


class Parameter:
    '''
    The Parameter class contains the following properties:

    * value: a float whose default value is 0
    * sigma: a float whose default value is 0
    * mutable: a boolean whose default value is False
    * constraints: an instance of the Constraints class, by default it is the default Constraints
    '''

    def __init__(self, value=0, sigma=0, mutable=False, constraints=Constraints()):
        try:
            self.value = float(value)
            self.sigma = float(sigma)
        except:
            raise ValueError("non-number value creeping into param" + str(value) + " " + str(sigma))
        self.mutable = mutable
        self.constraints = constraints

    @property
    def isConstrained(self):
        '''
        check if there are constrains. Return True is there is at least on constrain value.

        :return: True
        '''
        if self.constraints.link != -1:
            return True
        if self.constraints.max_index != -1:
            return True
        if self.constraints.MaxValue != math.inf:
            return True
        if self.constraints.min_index != -1:
            return True
        if self.constraints.MinValue != -math.inf:
            return True
        return False

    def serialize(self):
        """
        saves the contents of a class to a dictionary. unlike other serialize methods, not used in creating ParamterTree
        to send to D+ Calculation. Serialized parameters are expected by D+ as a *result* of fitting.

        :return: dictionary of the class fields (Value, isMutable, consMinIndex,consMaxIndex, linkIndex, sigma and constraints)
        """
        return {"Value": self.value,
                "isMutable": self.mutable,
                "isConstrained": self.isConstrained,
                "consMin": self.constraints.MinValue,
                "consMax": self.constraints.MaxValue,
                "consMinIndex": self.constraints.min_index,
                "consMaxIndex": self.constraints.max_index,
                "linkIndex": self.constraints.link,
                "sigma": self.sigma}

    def __str__(self):
        return str(self.serialize())

    def __repr__(self):
        return str(self.serialize())


class Model:
    '''
    A base class to D+ models.
    '''
    _model_ptr_index = 0

    def __init__(self):
        self.name = ""
        self.use_grid = True
        self.model_ptr = Model._model_ptr_index
        Model._model_ptr_index += 1

        self.extra_params = {}
        self.extra_param_index_map = []
        self.location_params = {}
        self.location_param_index_map = ["x", "y", "z", "alpha", "beta", "gamma"]

        self._init_from_metadata()

    def _init_from_metadata(self):
        # location params:
        location_vals = ["x", "y", "z", "alpha", "beta", "gamma"]
        for val in location_vals:
            self.location_params[val] = Parameter()

        # extra params:
        try:
            e_params = self.metadata["extraParams"]
        except:  # nothing to do here
            return
        for index, param in enumerate(e_params):
            self.extra_param_index_map.append(param["name"])
            self.extra_params[param["name"]] = Parameter(value=param["defaultValue"])

    def serialize(self):
        """
        saves the contents of a class to a dictionary.

        :return: dictionary of the class fields.
        """
        mydict = {"ModelPtr": self.model_ptr, "Name": self.name, "Use_Grid": self.use_grid,
                  "nExtraParams": len(self.extra_params),
                  "nLayers": 0, "nlp": 0,  # this is default, overwritten by modelWithLayers
                  "Type": _int_to_type(
                      # for now, type must be proceeded with comma because we haven't gotten rid of containers yet
                      self.index),  # self.index is set in the factory
                  "Mutables": [],
                  "Parameters": [],
                  "Sigma": [],
                  "Constraints": [],
                  "ExtraParameters": [], "ExtraConstraints": [], "ExtraMutables": [], "ExtraSigma": [],
                  "Location": {}, "LocationConstraints": {}, "LocationMutables": {}, "LocationSigma": {}
                  }
        # extraparams
        for i, param_name in enumerate(self.extra_param_index_map):
            param = self.extra_params[param_name]
            mydict["ExtraParameters"].append(param.value)
            mydict["ExtraConstraints"].append(param.constraints.serialize())
            mydict["ExtraMutables"].append(param.mutable)
            mydict["ExtraSigma"].append(param.sigma)

        # locationparams
        for param_name in self.location_params:
            param = self.location_params[param_name]
            mydict["Location"][param_name] = param.value
            mydict["LocationConstraints"][param_name] = param.constraints.serialize()
            mydict["LocationMutables"][param_name] = param.mutable
            mydict["LocationSigma"][param_name] = param.sigma

        return mydict

    def __str__(self):
        return (str(self.serialize()))

    def load_from_dictionary(self, json):
        '''
        sets the values of the various fields within a class to match those contained within a suitable dictionary.

        :param json:  json dictionary
        '''

        # first, check that the type matches the model's type index and everything is in order
        # Domains and populations don't have metadata, their type_index is -1, skip this section
        if self.index == -1:
            pass
        else:
            type_index = _type_to_int(json["Type"])

            if type_index != self.index:
                raise ValueError("Model type index mismatch")

        # override instance values
        try:
            self.name = json["Name"]
        except KeyError:
            pass  # we don't require names
        self.model_ptr = json["ModelPtr"]
        self.use_grid = json.get("Use_Grid", False)

        for param_index in range(len(json.get("ExtraParameters", []))):
            param = Parameter(value=json["ExtraParameters"][param_index], mutable=json["ExtraMutables"][param_index],
                              sigma=json["ExtraSigma"][param_index],
                              constraints=Constraints.from_dictionary(json["ExtraConstraints"][param_index]))
            self.extra_params[self.extra_param_index_map[param_index]] = param

        for param_index in json.get("Location", []):
            param = Parameter(value=json["Location"][param_index], mutable=json["LocationMutables"][param_index],
                              sigma=json["LocationSigma"][param_index],
                              constraints=Constraints.from_dictionary(json["LocationConstraints"][param_index]))
            self.location_params[param_index] = param

    def get_mutable_params(self):
        '''
        used in combining fitting results, or running fitting from within python

        :return: returns all the mutables params in extra_params and location_params
        '''
        mut_array = []
        # location params
        for param_name in self.location_param_index_map:
            if self.location_params[param_name].mutable:
                mut_array.append(self.location_params[param_name])

        # mutable params
        for param_name in self.extra_param_index_map:
            if self.extra_params[param_name].mutable:
                mut_array.append(self.extra_params[param_name])

        return mut_array

    def set_mutable_params(self, mut_arr):
        '''
        receives an order array of mutable params and set the values in extra_params and location_params according to that array

        :param mut_arr: array of mutable params
        '''
        param_index = 0
        for param_name in self.location_param_index_map:
            if self.location_params[param_name].mutable:
                self.location_params[param_name].value = mut_arr[param_index]
                param_index += 1
        for param_name in self.extra_param_index_map:
            if self.extra_params[param_name].mutable:
                self.extra_params[param_name].value = mut_arr[param_index]
                param_index += 1

            # location params

    def __basic_json_params(self):
        '''

        :return: a dictionary in the form:
        {
            "ModelPtr": self.model_ptr,
            "Parameters": params,
            "Submodels": []
        }

        submodels contains an array of this exact dictionary for child models.
        Parameters is an array of parameters, always in the following order:

        * x
        * y
        * z
        * alpha
        * beta
        * gamma
        * useGrid
        * number of layers
        * params[i][j]
        ...
        ...
        * extraparams[i]
        ...
        '''
        params = []
        # add default location params
        # add location params
        location_vals = ["x", "y", "z", "alpha", "beta", "gamma"]
        for val in location_vals:
            try:
                params.append(self.location_params[val].serialize())
            except:  # if we don't have location params, no big, just attach defaults
                params.append(Parameter().serialize())

        # add useGrid
        if self.use_grid:
            params.append(Parameter(1).serialize())
        else:
            params.append(Parameter(0).serialize())

        # add number of layers
        params.append(Parameter(1).serialize())

        # add extra params
        for param in self.extra_param_index_map:
            params.append(self.extra_params[param].serialize())

        return {
            "ModelPtr": self.model_ptr,
            "Parameters": params,
            "Submodels": []
        }


class ModelWithChildren(Model):
    '''
    D+ has few models which can have children. For example: Domain, population and Symmetry models
    '''

    def __init__(self):
        self.Children = []
        super().__init__()

    def serialize(self):
        '''
        saves the contents of a class to a dictionary.

        :return: dictionary of the class fields.
        '''

        mydict = super().serialize()

        mydict.update(
            {
                "Children": [child.serialize() for child in self.Children]
            }
        )
        return mydict

    def __str__(self):
        return (str(self.serialize()))

    def load_from_dictionary(self, json):
        '''
         sets the values of the various fields within a class to match those contained within a suitable dictionary.

         :param json:  json dictionary
         '''

        super().load_from_dictionary(json)
        for child in json["Children"]:
            childmodel = ModelFactory.create_model_from_dictionary(child)
            self.Children.append(childmodel)

    def __basic_json_params(self):
        basic_dict = super().__basic_json_params()
        for child in self.Children:
            basic_dict["Submodels"].append(child.__basic_json_params())
        return basic_dict


class ModelWithLayers(Model):
    '''
    D+ has few models which can have layers. For example: Sphere, Helix and UniformHollowCylinder
    '''

    def __init__(self):
        self.layer_params = []
        super().__init__()

    def _init_from_metadata(self):
        super()._init_from_metadata()
        # layer params:
        layerinfo = self.metadata["layers"]["layerInfo"]
        params = self.metadata["layers"]["params"]
        for layer in layerinfo:
            layer_dict = {}
            for param_index, parameter in enumerate(params):
                if layer["index"] == -1:
                    # This is just an indication of the default layer when more layers are added
                    # it is not an actual layer.
                    self._default_layer = Parameter(value=layer["defaultValues"][param_index])
                else:
                    layer_dict[parameter] = Parameter(value=layer["defaultValues"][param_index])
            if layer["index"] != -1:
                self.layer_params.append(layer_dict)
        self.layer_param_index_map = self.metadata["layers"]["params"]

    def parameters_to_json_arrays(self):
        json_dict = {"Parameters": [], "Constraints": [], "Mutables": [], "Sigma": []}
        # layerparams
        for layer in self.layer_params:
            param_array = []
            constr_array = []
            mut_array = []
            sigma_array = []
            for i, param_name in enumerate(self.layer_param_index_map):
                param = layer[param_name]
                param_array.append(param.value)
                constr_array.append(param.constraints.serialize())
                mut_array.append(param.mutable)
                sigma_array.append(param.sigma)
            json_dict["Parameters"].append(param_array)
            json_dict["Constraints"].append(constr_array)
            json_dict["Mutables"].append(mut_array)
            json_dict["Sigma"].append(sigma_array)

        # some additional things that are necessary
        json_dict["nlp"] = len(self.layer_params[0])
        json_dict["nLayers"] = len(self.layer_params)
        return json_dict

    def load_from_dictionary(self, json):
        '''
         sets the values of the various fields within a class to match those contained within a suitable dictionary.

         :param json:  json dictionary
         '''
        super().load_from_dictionary(json)
        for layer_index in range(len(json["Parameters"])):
            for param_index in range(len(json["Parameters"][layer_index])):
                param = Parameter(value=json["Parameters"][layer_index][param_index],
                                  mutable=json["Mutables"][layer_index][param_index],
                                  sigma=json["Sigma"][layer_index][param_index],
                                  constraints=Constraints.from_dictionary(
                                      json["Constraints"][layer_index][param_index]))
                try:
                    self.layer_params[layer_index][self.layer_param_index_map[param_index]] = param
                except IndexError:
                    if len(json["Parameters"]) > self.metadata["layers"]["max"] and self.metadata["layers"][
                        "max"] != -1:
                        raise ValueError(
                            "Not allowed to set more than " + str(self.metadata["layers"]["max"]) + " layers")

                    # otherwise go ahead and add the layer
                    self.layer_params.append({})
                    self.layer_params[layer_index][self.layer_param_index_map[param_index]] = param

    def serialize(self):
        '''
         saves the contents of a class to a dictionary.

         :return: dictionary of the class fields.
         '''
        mydict = super().serialize()

        mydict.update(
            self.parameters_to_json_arrays()
        )
        return mydict

    def get_mutable_params(self):
        '''
        Return all the mutable params of the model. The mutable params come from the layers array,  extra_params and location_params.

        :return: mutable params array
        '''
        mut_array = []

        # location params
        for param_name in self.location_param_index_map:
            if self.location_params[param_name].mutable:
                mut_array.append(self.location_params[param_name])
        # layer params
        for layer in self.layer_params:
            for param_name in self.layer_param_index_map:
                if layer[param_name].mutable:
                    mut_array.append(layer[param_name])
        # extra params
        for param_name in self.extra_param_index_map:
            if self.extra_params[param_name].mutable:
                mut_array.append(self.extra_params[param_name])

        return mut_array

    def set_mutable_params(self, mut_array):
        '''
          receives an order array of mutable params and set the values in layer , extra_params and location_params according to that array.

          :param mut_arr: array of mutable params
          '''
        index = 0

        # location params
        for param_name in self.location_param_index_map:
            if self.location_params[param_name].mutable:
                self.location_params[param_name] = mut_array[index]
                index += 1

        # layer params
        for layer in self.layer_params:
            for param_name in layer:
                if layer[param_name].mutable:
                    layer[param_name] = mut_array[index]
                    index += 1

        # extra params
        for param_name in self.extra_param_index_map:
            if self.extra_params[param_name].mutable:
                self.extra_params[param_name] = mut_array[index]
                index += 1

    def __basic_json_params(self):
        '''
        :param use_grid:
        :return:
        x
        y
        z
        alpha
        beta
        gamma
        useGrid
        number of layers
        params[i][j]
        ...
        ...
        extraparams[i]
        ...
        '''
        # basic_dict = super().__basic_json_params(useGrid)
        # override basic entirely
        basic_dict = super().__basic_json_params()
        super_params_arr = basic_dict["Parameters"]

        # the first 7 params are location and use_grid and remain unchanged. The rest are overwritten
        params = super_params_arr[:7]

        # add number of layers
        params.append(Parameter(len(self.layer_params)).serialize())

        # add params:
        for param in self.layer_param_index_map:
            for layer in self.layer_params:
                params.append(layer[param].serialize())

        # add extra params
        for param in self.extra_param_index_map:
            params.append(self.extra_params[param].serialize())

        basic_dict["Parameters"] = params
        return basic_dict


class ModelWithFile(Model):
    '''
    D+ has few models which have a file. For example: PDB, AMP and ScriptedSymmetry
    '''

    def __init__(self, filename=""):
        self.filenames = []
        self.filename = filename
        super().__init__()

    def serialize(self):
        '''
         saves the contents of a class to a dictionary.

         :return: dictionary of the class fields.
         '''
        mydict = super().serialize()

        mydict.update(
            {
                "Filename": self.filename,
            }
        )

        try:
            mydict.update(
                {
                    "Centered": self.centered,
                }
            )
        except (AttributeError, KeyError) as err:  # not everything has centered
            pass

        try:
            mydict.update(
                {
                    "AnomFilename": self.anomfilename,
                }
            )
        except (AttributeError, KeyError) as err:  # not everything has an anomfilename
            pass
        return mydict

    def __str__(self):
        return (str(self.serialize()))

    def load_from_dictionary(self, json):
        '''
         sets the values of the various fields within a class to match those contained within a suitable dictionary.

         :param json:  json dictionary
         '''
        super().load_from_dictionary(json)
        self.filename = json["Filename"]
        self.filenames.append(self.filename)

        # TODO: various optional additonal fields that really should be handled in a better way
        try:
            self.centered = json["Centered"]
        except (AttributeError, KeyError) as err:  # not everything has centered
            pass

        try:
            self.anomfilename = json["AnomFilename"]
            self.filenames.append(json["AnomFilename"])
        except KeyError as err:  # not everything has an anomfilename
            pass


def _get_model_tuple(metadata):
    model_list = []
    try:
        if len(metadata["layers"]["layerInfo"]) > 0:
            # if metadata["isLayerBased"] == True:
            model_list.append(ModelWithLayers)
    except:
        pass
    if metadata["category"] == 9:  # symmetry
        model_list.append(ModelWithChildren)
    if metadata["name"] in _models_with_files_index_dict:
        model_list.append(ModelWithFile)
    if len(model_list) == 0:
        model_list = [Model]
    return tuple(model_list)


class ScriptedSymmetry(Model):
    '''
    A class for D+ ScriptedSymmetry, this is sufficient for running against existing backend,\
     but does NOT implement running with python fit
    '''

    # TODO: this is sufficient for running against existing backend, but does NOT implement running with python fit
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def load_from_dictionary(self, json):
        '''
         sets the values of the various fields within a class to match those contained within a suitable dictionary.

         :param json:  json dictionary
         '''
        # print(vars(self))
        self.__dict__.update(**json)
        self.json = json
        if "Filename" in json:
            self.filename = json["Filename"]
            self.filenames = [self.filename]
        if "Children" in json:
            self.Children = []
            for child in json["Children"]:
                childmodel = ModelFactory.create_model_from_dictionary(child)
                self.Children.append(childmodel)

    def serialize(self):
        '''
         saves the contents of a class to a dictionary.

         :return: dictionary of the class fields.
         '''
        return_dict = {}
        for key in self.json:
            return_dict[key] = self.__dict__[key]

        if "Children" in self.json:
            return_dict["Children"] = [child.serialize() for child in self.Children]

        return return_dict


class ModelFactory:
    models_arr = []
    from types import ModuleType
    models = ModuleType('dplus.DataModels.models')
    sys.modules['dplus.DataModels.models'] = models

    @classmethod
    def add_model(cls, metadata):
        no_space_name = "".join(metadata["name"].split())
        no_space_name = no_space_name.replace("-", "")
        modeltuple = _get_model_tuple(metadata)

        # replace name with type_name
        metadata["type_name"] = metadata.pop("name")
        metadata["metadata"] = metadata.copy()
        myclass = type(no_space_name, modeltuple,
                       metadata)

        ModelFactory.models_arr.append(myclass)
        setattr(ModelFactory.models, no_space_name, myclass)

    @classmethod
    def create_model_from_dictionary(cls, json):
        model_index_str = json["Type"]
        if model_index_str in ["Scripted Geometry", "Scripted Model"]:
            raise NotImplemented(
                "Tal says:Scripted models and geometries are remnants of a yet unimplemented feature (script models, e.g., written in Python). They should be obliterated from existence for now, only to be revived if python models work.")

        if model_index_str == "Scripted Symmetry":
            m = ScriptedSymmetry(**json)
            m.load_from_dictionary(json)
            return m

        model_index = _type_to_int(model_index_str)

        for model in ModelFactory.models_arr:  # TODO: Turn this into a dictionary at some point
            if model.index == model_index:
                m = model()
                m.load_from_dictionary(json)
                return m

        raise ValueError("Model not found")

    @classmethod
    def create_model(cls, name_or_index):
        no_space_name = "_".join(name_or_index.split())
        for model in ModelFactory.models_arr:
            if model.type_index == name_or_index or model.type_name == name_or_index or model.type_name == no_space_name:
                return model

        raise ValueError("Model not found")


class Population(ModelWithChildren):
    '''
    `Population` can contain a number of `Model` classes. Some models have children, which are also models.
    '''
    index = -1

    def __init__(self):
        super().__init__()
        self.population_size = 1
        self.population_size_mut = False
        self.extra_param_index_map = ["Population Size"]
        self.extra_params["Population Size"] = Parameter(value=self.population_size,
                                                         mutable=self.population_size_mut)

    @property
    def models(self):
        '''
        Return all the models in the population class.

        :return: models array
        '''
        return self.Children

    def add_model(self, model):
        '''

        :param model: model to add to the population
        '''
        self.models.append(model)

    def serialize(self):
        """
          saves the contents of a class Population to a dictionary.

          :return: dictionary of the class fields.
        """
        mydict = super().serialize()
        mydict["Models"] = mydict.pop("Children")

        newdict = {
            "PopulationSize": self.population_size,
            "PopulationSizeMut": self.population_size_mut,
            "ModelPtr": self.model_ptr,
            "Models": mydict["Models"]
        }

        return newdict

    def load_from_dictionary(self, json):
        '''
        sets the values of the various fields within a class to match those contained within a suitable dictionary.

        :param json: json dictionary
        '''
        self.model_ptr = json["ModelPtr"]
        for model in json["Models"]:
            self.Children.append(ModelFactory.create_model_from_dictionary(model))
        self.population_size = json["PopulationSize"]
        self.population_size_mut = json["PopulationSizeMut"]
        self.extra_params["Population Size"] = Parameter(value=self.population_size,
                                                         mutable=self.population_size_mut)


class Domain(ModelWithChildren):
    '''
    The Domain class describes the parameter tree.
    The Domain model is the root of the parameter tree, which can contain multiple populations.
    '''
    index = -1

    def __init__(self):
        super().__init__()
        self.scale = 1
        self.constant = 0.0
        self.scale_mut = False
        self.constant_mut = False
        self.geometry = "Domains"
        self.populations.append(Population())
        self.extra_param_index_map = ["Scale", "Constant"]
        self.extra_params["Constant"] = Parameter(value=self.constant, mutable=self.constant_mut)
        self.extra_params["Scale"] = Parameter(value=self.scale, mutable=self.scale_mut)

    @property
    def populations(self):
        '''

        :return: The populations of the domain
        '''
        return self.Children

    def serialize(self):
        """
          saves the contents of a class Domain to a dictionary.

          :return: dictionary of the class fields.
        """

        # we need to completely override the dictionary returned by model
        # (which includes nlayers and other extraneous fields).

        mydict = super().serialize()
        mydict["Populations"] = mydict.pop("Children")

        newdict = {
            "ModelPtr": self.model_ptr,
            "Scale": self.scale,
            "ScaleMut": self.scale_mut,
            "Constant": self.constant,
            "ConstantMut": self.constant_mut,
            "Geometry": self.geometry,
            "Populations": mydict["Populations"]
        }

        return newdict

    def load_from_dictionary(self, json):
        """
        sets the values of the various fields within a class to match those contained within a suitable dictionary.

        :param json: json dictionary
        """
        self.populations[
        :] = []  # by default Domain creates an empty population. However if we are loading from json we don't want this empty population
        for population in json["Populations"]:
            popu = Population()
            popu.load_from_dictionary(population)
            self.Children.append(popu)
        self.scale = json["Scale"]
        self.scale_mut = json["ScaleMut"]
        try:
            self.constant = json["Constant"]  # TODO: add back if necessary
            self.constant_mut = json["ConstantMut"]
        except Exception as e:
            print(e)  # is probably an old model without constant
        self.geometry = json["Geometry"]
        self.model_ptr = json["ModelPtr"]
        self.extra_params["Constant"] = Parameter(value=self.constant, mutable=self.constant_mut)
        self.extra_params["Scale"] = Parameter(value=self.scale, mutable=self.scale_mut)

    def __basic_json_params(self, useGrid):
        '''

        :param useGrid:
        :return:
        '''
        self.use_grid = useGrid
        basic_dict = super().__basic_json_params(useGrid)
        # we need to add in parameters to the domain
        basic_dict["Parameters"].append(self.scale_param.serialize())
        for population in self.Children:
            basic_dict["Parameters"].append(population.population_size_param.serialize())
        return basic_dict


for model in hardcode_models:
    ModelFactory.add_model(model)

for model in meta_models:
    ModelFactory.add_model(model)
