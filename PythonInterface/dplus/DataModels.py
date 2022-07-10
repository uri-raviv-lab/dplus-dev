import math
import sys
import csv
from pathlib import Path
from collections.abc import MutableSequence
from collections import UserDict
from copy import deepcopy
from dplus.FileReaders import _handle_infinity_for_json
from dplus.metadata import meta_models, hardcode_models, _type_to_int, _models_with_files_index_dict, _int_to_type


def make_name_pythonic(name, is_cls=False):
    if is_cls:
        join_char = ""
    else:
        join_char = "_"
        
    no_space_name = join_char.join(name.split())
    no_space_name = no_space_name.replace("-", join_char)
    no_space_name = no_space_name.replace(".", join_char)
    if is_cls:
        return no_space_name
    
    return no_space_name.lower()

class Constraints:
    '''
    The Constraints class contains the following properties:

    * max_value: a float whose default value is infinity
    * min_value: a float whose default value is -infinity
    '''

    def __init__(self, max_val=math.inf, min_val=-math.inf, minindex=-1, maxindex=-1, link=-1):
        if max_val == "inf":
            max_val = math.inf
        if min_val == "-inf":
            min_val = -math.inf

        if max_val <= min_val:
            raise ValueError("Constraints' upper bound must be greater than lower bound")

        self.max_value = max_val
        self.min_value = min_val
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
        return_dict= _handle_infinity_for_json({
            "Link": self.link,
            "MaxIndex": self.max_index,
            "MaxValue": self.max_value,
            "MinIndex": self.min_index,
            "MinValue": self.min_value
        })
        return return_dict


class Parameter:
    '''
    The Parameter class contains the following properties:

    * value: a float whose default value is 0
    * sigma: a float whose default value is 0
    * mutable: a boolean whose default value is False
    * constraints: an instance of the Constraints class, by default it is the default Constraints
    '''

    def __init__(self, value=0, sigma=0, mutable=False, constraints=Constraints(), name=""):
        try:
            self.value = float(value)
            self.sigma = float(sigma)
        except:
            raise ValueError("non-number value creeping into param" + str(value) + " " + str(sigma))
        self.mutable = mutable
        self.constraints = constraints
        self.name = name

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
        if self.constraints.max_value != math.inf:
            return True
        if self.constraints.min_index != -1:
            return True
        if self.constraints.min_value != -math.inf:
            return True
        return False

    def serialize(self):
        """
        saves the contents of a class to a dictionary. unlike other serialize methods, not used in creating ParamterTree
        to send to D+ Calculation. Serialized parameters are expected by D+ as a *result* of fitting.

        :return: dictionary of the class fields (Value, isMutable, consMinIndex,consMaxIndex, linkIndex, sigma and constraints)
        """
        return _handle_infinity_for_json({"Value": self.value,
                "isMutable": self.mutable,
                "isConstrained": self.isConstrained,
                "consMin": self.constraints.min_value,
                "consMax": self.constraints.max_value,
                "consMinIndex": self.constraints.min_index,
                "consMaxIndex": self.constraints.max_index,
                "linkIndex": self.constraints.link,
                "sigma": self.sigma,
                "name": self.name
                })

    def __str__(self):
        return str(self.serialize())

    def __repr__(self):
        return str(self.serialize())


class ParameterContainer(UserDict):
    def __init__(self, data):
        self.__dict__.update(deepcopy(data))
    @property
    def data(self):
        return self.__dict__
    def __setitem__(self, key, item):
        if key not in self.data:
            raise KeyError("The parameter {} is not defined.".format(key))
        if not isinstance(item, Parameter):
            raise ValueError("{} can only be set to an instance of Parameter".format(item))
        self.data[key] = item

    def __delitem__(self, k):
        raise ValueError("You cannot delete a parameter from a layer")


class Layers(MutableSequence):
    def __init__(self, data=[], min_length=0, max_length=-1):
        self.list = list()
        self.extend(list(data))
        self.min_length = min_length
        if max_length == -1:
            max_length = math.inf
        self.max_length = max_length

    def __setitem__(self, i, item):
        if not isinstance(item, ParameterContainer):
            raise ValueError("You can't add an invalid layer")
        self.list[i] = item

    def __delitem__(self, i):
        if len(self.list) - 1 < self.min_length:
            raise ValueError("This model doesn't allow fewer than {} layers".format(self.max_length))
        del self.list[i]

    def __getitem__(self, i):
        return self.list[i]

    def insert(self, i, item):
        if not isinstance(item, ParameterContainer):
            raise ValueError("You can't add an invalid layer")
        if len(self.list) + 1 > self.max_length:
            raise ValueError("This model doesn't allow more than {} layers".format(self.max_length))
        self.list.insert(i, item)

    def __str__(self):
        return str(self.list)

    def __len__(self):
        return len(self.list)


class Children(MutableSequence):
    def __init__(self, data=[]):
        self.list = list()
        self.extend(list(data))

    def __setitem__(self, i, item):
        if not isinstance(item, Model) and item != []:
            raise ValueError("A model's children must be models")
        self.list[i] = item

    def __delitem__(self, i):
        del self.list[i]

    def __getitem__(self, i):
        return self.list[i]

    def insert(self, i, item):
        self.list.insert(i, item)

    def __str__(self):
        return str(self.list)

    def __len__(self):
        return len(self.list)


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
        self._extra_param_index_map = []
        self.location_params = {}
        self._location_param_index_map = ["x", "y", "z", "alpha", "beta", "gamma"]

        self._init_from_metadata()

    def _init_from_metadata(self):
        # location params:
        location_vals = ["x", "y", "z", "alpha", "beta", "gamma"]
        location_params_dict = {}
        for val in location_vals:
            location_params_dict[val] = Parameter(name=val)
        self.location_params = ParameterContainer(location_params_dict)
        # extra params:
        try:
            e_params = self._metadata["extraParams"]
        except:  # nothing to do here
            return
        extra_params_dict = {}
        for index, param in enumerate(e_params):
            p_name = make_name_pythonic(param["name"])
            self._extra_param_index_map.append(p_name)
            extra_params_dict[p_name] = Parameter(value=param["defaultValue"], name=p_name)
        self.extra_params = ParameterContainer(extra_params_dict)

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
                      self._metadata["index"]),  # self.index is set in the factory
                  "Mutables": [],
                  "Parameters": [],
                  "Sigma": [],
                  "Constraints": [],
                  "ExtraParameters": [], "ExtraConstraints": [], "ExtraMutables": [], "ExtraSigma": [],
                  "Location": {}, "LocationConstraints": {}, "LocationMutables": {}, "LocationSigma": {}
                  }
        # extraparams
        for i, param_name in enumerate(self._extra_param_index_map):
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
        if self._metadata["index"] == -1:
            pass
        else:
            type_index = _type_to_int(json["Type"])

            if type_index != self._metadata["index"]:
                raise ValueError("Model type index mismatch")

        # override instance values
        try:
            self.name = json["Name"]
        except KeyError:
            pass  # we don't require names
        self.model_ptr = json["ModelPtr"]
        self.use_grid = json.get("Use_Grid", False)

        for param_index in range(len(json.get("ExtraParameters", []))):
            param_name = self._extra_param_index_map[param_index]
            param = Parameter(value=json["ExtraParameters"][param_index], mutable=json["ExtraMutables"][param_index],
                              sigma=json["ExtraSigma"][param_index],
                              constraints=Constraints.from_dictionary(json["ExtraConstraints"][param_index]),
                              name=param_name)
            self.extra_params[param_name] = param

        for param_name in json.get("Location", []):
            param = Parameter(value=json["Location"][param_name], mutable=json["LocationMutables"][param_name],
                              sigma=json["LocationSigma"][param_name],
                              constraints=Constraints.from_dictionary(json["LocationConstraints"][param_name]),
                              name=param_name)
            self.location_params[param_name] = param

    def get_mutable_params(self):
        '''
        used in combining fitting results, or running fitting from within python

        :return: returns all the mutables params in extra_params and location_params
        '''
        mut_array = []
        # location params
        for param_name in self._location_param_index_map:
            if self.location_params[param_name].mutable:
                mut_array.append(self.location_params[param_name])

        # mutable params
        for param_name in self._extra_param_index_map:
            if self.extra_params[param_name].mutable:
                mut_array.append(self.extra_params[param_name])

        return mut_array

    def set_mutable_params(self, mut_arr):
        '''
        receives an order array of mutable params and set the values in extra_params and location_params according to that array

        :param mut_arr: array of mutable params
        '''
        param_index = 0

        for param_name in self._location_param_index_map:
            if self.location_params[param_name].mutable:
                self.location_params[param_name].value = mut_arr[param_index]
                param_index += 1

        for param_name in self._extra_param_index_map:
            if self.extra_params[param_name].mutable:
                self.extra_params[param_name].value = mut_arr[param_index]
                param_index += 1
        if type(self) == Domain:
            self.constant = self.extra_params['constant'].value
            self.scale = self.extra_params['scale'].value
            # location params

    def _basic_json_params(self):
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
                params.append(Parameter(name=val).serialize())

        # add useGrid
        if self.use_grid:
            params.append(Parameter(1, name="use_grid").serialize())
        else:
            params.append(Parameter(0, name="use_grid").serialize())

        # add number of layers
        params.append(Parameter(1, name="numlayers").serialize())

        # add extra params
        for param in self._extra_param_index_map:
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
        self.children = Children()
        super().__init__()

    def serialize(self):
        '''
        saves the contents of a class to a dictionary.

        :return: dictionary of the class fields.
        '''

        mydict = super().serialize()

        mydict.update(
            {
                "Children": [child.serialize() for child in self.children]
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
            self.children.append(childmodel)

    def _basic_json_params(self):
        basic_dict = super()._basic_json_params()
        for child in self.children:
            basic_dict["Submodels"].append(child._basic_json_params())
        return basic_dict


class ModelWithLayers(Model):
    '''
    D+ has few models which can have layers. For example: Sphere, Helix and UniformHollowCylinder
    '''

    def __init__(self):
        super().__init__()

    @property
    def default_layer(self):
        _default_layer={}
        layer = self._metadata["layers"]["layerInfo"][-1]
        for param_index, parameter in enumerate(self._layer_param_index_map):
            _default_layer[parameter] = Parameter(value=layer["defaultValues"][param_index],
                                                   name=parameter)
        return ParameterContainer(_default_layer)

    def _init_from_metadata(self):
        super()._init_from_metadata()
        # layer params:
        layerinfo = self._metadata["layers"]["layerInfo"]
        self._layer_param_index_map = []
        for p_name in self._metadata["layers"]["params"]:
            self._layer_param_index_map.append(make_name_pythonic(p_name))

        self.layer_params = Layers(min_length=self._metadata["layers"]["min"],
                                   max_length=self._metadata["layers"]["max"])
        for layer in layerinfo:
            if layer["index"] == -1:
                continue
            layer_dict = {}
            for param_index, parameter in enumerate(self._layer_param_index_map):
                layer_dict[parameter] = Parameter(value=layer["defaultValues"][param_index], name=parameter)
            self.layer_params.append(ParameterContainer(layer_dict))

    def parameters_to_json_arrays(self):
        json_dict = {"Parameters": [], "Constraints": [], "Mutables": [], "Sigma": []}
        # layerparams
        for layer in self.layer_params:
            param_array = []
            constr_array = []
            mut_array = []
            sigma_array = []
            for i, param_name in enumerate(self._layer_param_index_map):
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
            layer_dict = {}
            for param_index in range(len(json["Parameters"][layer_index])):
                param_name = self._layer_param_index_map[param_index]
                param = Parameter(value=json["Parameters"][layer_index][param_index],
                                  mutable=json["Mutables"][layer_index][param_index],
                                  sigma=json["Sigma"][layer_index][param_index],
                                  constraints=Constraints.from_dictionary(
                                      json["Constraints"][layer_index][param_index]),
                                  name=param_name)
                layer_dict[param_name] = param
            try:
                self.layer_params[layer_index] = ParameterContainer(layer_dict)
            except IndexError:
                self.layer_params.append(ParameterContainer(layer_dict))
            except ValueError as e:
                hi = 1
                raise e

    def add_layer(self):
        self.layer_params.append(self.default_layer)
        return self.layer_params[-1]

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
        for param_name in self._location_param_index_map:
            if self.location_params[param_name].mutable:
                mut_array.append(self.location_params[param_name])
        # layer params
        for layer in self.layer_params:
            for param_name in self._layer_param_index_map:
                if layer[param_name].mutable:
                    mut_array.append(layer[param_name])
        # extra params
        for param_name in self._extra_param_index_map:
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
        for param_name in self._location_param_index_map:
            if self.location_params[param_name].mutable:
                self.location_params[param_name].value = mut_array[index]
                index += 1

        # layer params
        for layer in self.layer_params:
            for param_name in layer:
                if layer[param_name].mutable:
                    layer[param_name].value = mut_array[index]
                    index += 1

        # extra params
        for param_name in self._extra_param_index_map:
            if self.extra_params[param_name].mutable:
                self.extra_params[param_name].value = mut_array[index]
                index += 1

    def _basic_json_params(self):
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
        # basic_dict = super()._basic_json_params(useGrid)
        # override basic entirely
        basic_dict = super()._basic_json_params()
        super_params_arr = basic_dict["Parameters"]

        # the first 7 params are location and use_grid and remain unchanged. The rest are overwritten
        params = super_params_arr[:7]

        # add number of layers
        params.append(Parameter(len(self.layer_params), name="nlp").serialize())

        # add params:
        for param in self._layer_param_index_map:
            for layer in self.layer_params:
                params.append(layer[param].serialize())

        # add extra params
        for param in self._extra_param_index_map:
            params.append(self.extra_params[param].serialize())

        basic_dict["Parameters"] = params
        return basic_dict


class ModelWithFile(Model):
    '''
    D+ has few models which have a file. For example: PDB, AMP and ScriptedSymmetry
    '''

    def __init__(self, filename=""):
        self.filenames = []
        self._filename = filename
        self._anomfilename=""
        super().__init__()

    @property
    def filename(self):
        return str(Path(self._filename).absolute())

    @filename.setter
    def filename(self, name):
        self._filename=name

    @property
    def anomfilename(self):
        if self._anomfilename:
            return str(Path(self._anomfilename).absolute())
        return ""

    @anomfilename.setter
    def anomfilename(self, name):
        if name:
            self._anomfilename = name
        else:
            self._anomfilename = ""



    def serialize(self):
        '''
         saves the contents of a class to a dictionary.

         :return: dictionary of the class fields.
         '''
        mydict = super().serialize()

        mydict.update(
            {
                "Filename": self.filename,
                "AnomFilename": self.anomfilename,
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
            self.filenames.append(self.anomfilename)
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
            self.children = []
            for child in json["Children"]:
                childmodel = ModelFactory.create_model_from_dictionary(child)
                self.children.append(childmodel)

    def get_mutable_params(self):
        mut_array = []

        # location params
        if hasattr(self, 'Location'):
            location_vals = ["x", "y", "z", "alpha", "beta", "gamma"]
            for param_name in location_vals:
                if self.LocationMutables[param_name]:
                    mut_array.append(Parameter(value=self.Location[param_name],
                                               sigma=self.LocationSigma[param_name],
                                               mutable=self.LocationMutables[param_name],
                                               constraints=Constraints.from_dictionary(
                                                   self.LocationConstraints[param_name]),
                                               name=param_name))
        # layer params
        if hasattr(self, 'Parameters'):
            for param_val_list, sigma_list, mut_list, constraints_list in zip(self.Parameters, self.Sigma,
                                                                              self.Mutables, self.Constraints):
                for param_val, sigma, mut, constraints in zip(param_val_list, sigma_list, mut_list, constraints_list):
                    if mut:
                        mut_array.append(Parameter(value=param_val,
                                                   sigma=sigma,
                                                   mutable=mut,
                                                   constraints=Constraints.from_dictionary(constraints)))
        # extra params
        if hasattr(self, 'ExtraParameters'):
            for ex_param_val, ex_sigma, ex_mut, ex_constraints in zip(self.ExtraParameters, self.ExtraSigma,
                                                                      self.ExtraMutables, self.ExtraConstraints):
                if ex_mut:
                    mut_array.append(Parameter(value=ex_param_val,
                                               sigma=ex_sigma,
                                               mutable=ex_mut,
                                               constraints=Constraints.from_dictionary(ex_constraints)))
        return mut_array

    def set_mutable_params(self, mut_array):

        index = 0
        if hasattr(self, 'Location'):
            location_vals = ["x", "y", "z", "alpha", "beta", "gamma"]
            for param_name in location_vals:
                if self.LocationMutables[param_name]:
                    self.LocationMutables[param_name] = mut_array[index]
                    index += 1
        # layer params
        if hasattr(self, 'Parameters'):
            for list_index, (param_val_list, mut_list) in enumerate(zip(self.Parameters, self.Mutables)):
                for item_index, (param_val, mut) in enumerate(zip(param_val_list, mut_list)):
                    if mut:
                        self.Parameters[list_index][item_index] = mut_array[index]
                        index += 1
        # extra params
        if hasattr(self, 'ExtraParameters'):
            for cur_index, (extra_param_val, extra_mut) in enumerate(zip(self.ExtraParameters, self.ExtraMutables)):
                if extra_mut:
                    self.ExtraParameters[cur_index] = mut_array[index]
                    index += 1

    def serialize(self):
        '''
         saves the contents of a class to a dictionary.

         :return: dictionary of the class fields.
         '''
        return_dict = {}
        for key in self.json:
            return_dict[key] = self.__dict__[key]

        if "Children" in self.json:
            return_dict["Children"] = [child.serialize() for child in self.children]

        return return_dict

    def _basic_json_params(self):
        basic_dict = {
            "ModelPtr": self.ModelPtr,
            "Parameters": [],
            "Submodels": []
        }
        params = []
        if hasattr(self, 'Location'):
            location_vals = ["x", "y", "z", "alpha", "beta", "gamma"]
            for param_name in location_vals:
                params.append(Parameter(value=self.Location[param_name],
                                        sigma=self.LocationSigma[param_name],
                                        mutable=self.LocationMutables[param_name],
                                        constraints=Constraints.from_dictionary(self.LocationConstraints[param_name]),
                                        name=param_name).serialize())

        # add useGrid
        try:
            if self.Use_Grid:
                params.append(Parameter(1, name="UseGrid").serialize())
            else:
                params.append(Parameter(0, name="UseGrid").serialize())

        except:
            pass
            # add number of layers
        params.append(Parameter(self.nLayers, name="nLayers").serialize())

        if hasattr(self, 'Parameters'):
            for param_val_list, sigma_list, mut_list, constraints_list in zip(self.Parameters, self.Sigma,
                                                                              self.Mutables, self.Constraints):
                for param_val, sigma, mut, constraints in zip(param_val_list, sigma_list, mut_list, constraints_list):
                    params.append(Parameter(value=param_val,
                                            sigma=sigma,
                                            mutable=mut,
                                            constraints=Constraints.from_dictionary(constraints)).serialize())
        # extra params
        if hasattr(self, 'ExtraParameters'):
            for ex_param_val, ex_sigma, ex_mut, ex_constraints in zip(self.ExtraParameters, self.ExtraSigma,
                                                                      self.ExtraMutables, self.ExtraConstraints):
                params.append(Parameter(value=ex_param_val,
                                        sigma=ex_sigma,
                                        mutable=ex_mut,
                                        constraints=Constraints.from_dictionary(ex_constraints)).serialize())

        basic_dict["Parameters"] = params
        for child in self.children:
            basic_dict["Submodels"].append(child._basic_json_params())
        return basic_dict


class ModelFactory:
    models_arr = []
    from types import ModuleType
    models = ModuleType('dplus.DataModels.models')
    sys.modules['dplus.DataModels.models'] = models

    @classmethod
    def add_model(cls, metadata):
        no_space_name=make_name_pythonic(metadata["name"], is_cls=True)
        modeltuple = _get_model_tuple(metadata)

        # replace name with type_name
        metadata["type_name"] = metadata.pop("name")
        myclass = type(no_space_name, modeltuple,
                       {"_metadata": metadata})

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
            if model._metadata["index"] == model_index:
                m = model()
                m.load_from_dictionary(json)
                return m

        raise ValueError("Model not found")

    @classmethod
    def create_model(cls, name_or_index):
        no_space_name = make_name_pythonic(name_or_index, is_cls=True)
        for model in ModelFactory.models_arr:
            if model.type_index == name_or_index or model.type_name == name_or_index or model.type_name == no_space_name:
                return model

        raise ValueError("Model not found")


class Population(ModelWithChildren):
    '''
    `Population` can contain a number of `Model` classes. Some models have children, which are also models.
    '''
    _metadata = {"index": -1}

    def __init__(self):
        super().__init__()
        self.population_size = 1
        self.population_size_mut = False
        self.extra_param_index_map = ["population_size"]
        self.extra_params["population_size"] = Parameter(value=self.population_size,
                                                         mutable=self.population_size_mut,
                                                         name="population_size")

    @property
    def models(self):
        '''
        Return all the models in the population class.

        :return: models array
        '''
        return self.children

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
            self.children.append(ModelFactory.create_model_from_dictionary(model))
        self.population_size = json["PopulationSize"]
        self.population_size_mut = json["PopulationSizeMut"]
        self.extra_params["population_size"] = Parameter(value=self.population_size,
                                                         mutable=self.population_size_mut,
                                                         name="population_size")

    def _basic_json_params(self):
        '''

        :return:
        '''
        # for reasons unknown to any sane being, population size is treated as belonging to Domain, and is not expected
        # in the basic json params for population. since it is added as part of extra params, it is then politely removed, here

        res = super()._basic_json_params()
        res["Parameters"].pop()
        return res


class Domain(ModelWithChildren):
    '''
    The Domain class describes the parameter tree.
    The Domain model is the root of the parameter tree, which can contain multiple populations.
    '''
    _metadata = {"index": -1}

    def __init__(self):
        super().__init__()
        self.scale = 1
        self.constant = 0.0
        self.scale_mut = False
        self.constant_mut = False
        self.geometry = "Domains"
        self.populations.append(Population())
        self.extra_param_index_map = ["scale", "constant"]
        self.extra_params["constant"] = Parameter(value=self.constant, mutable=self.constant_mut, name="constant")
        self.extra_params["scale"] = Parameter(value=self.scale, mutable=self.scale_mut, name="scale")

    @property
    def populations(self):
        '''

        :return: The populations of the domain
        '''
        return self.children

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
            self.children.append(popu)
        self.scale = json["Scale"]
        self.scale_mut = json["ScaleMut"]
        try:
            self.constant = json["Constant"]  # TODO: add back if necessary
            self.constant_mut = json["ConstantMut"]
        except Exception as e:
            print(e)  # is probably an old model without constant
        self.geometry = json["Geometry"]
        self.model_ptr = json["ModelPtr"]
        self.extra_params["constant"] = Parameter(value=self.constant, mutable=self.constant_mut, name="constant")
        self.extra_params["scale"] = Parameter(value=self.scale, mutable=self.scale_mut, name="scale")

    def _basic_json_params(self, useGrid):
        '''

        :param useGrid:
        :return:
        '''
        self.use_grid = useGrid
        basic_dict = super()._basic_json_params()
        # we need to add in parameters to the domain
        for population in self.children:
            basic_dict["Parameters"].append(population.extra_params["population_size"].serialize())

        return basic_dict


class ManualSymmetry(ModelWithChildren, ModelWithLayers):
    '''
    A class for D+ ManualSymmetry
    '''
    _metadata = {
                "index": 26,
                "type_name": "Manual Symmetry",
                "category": 9,
                "gpuCompatible": False,
                "slow": False,
                "ffImplemented": False,
                "isLayerBased": True,
                "layers": {
                    "min": 0,
                    "max": -1,
                    "layerInfo": [
                        {
                            "index": -1,
                            "name": "Instance %d",
                            "applicability": [
                                1,
                                1,
                                1,
                                1,
                                1,
                                1
                            ],
                            "defaultValues": [
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0
                            ]
                        }
                    ],
                    "params": [
                        "X",
                        "Y",
                        "Z",
                        "Alpha",
                        "Beta",
                        "Gamma"
                    ]
                },
                "extraParams": [
                    {
                        "name": "Scale",
                        "defaultValue": 1.0,
                        "isIntegral": False,
                        "decimalPoints": 12,
                        "isAbsolute": False,
                        "canBeInfinite": False
                    }
                ],
        "modelCategories": {
                "name": "Symmetries",
                "index": 9,
                "type": 8,
                "models": [
                    25,
                    26
                ]
            }
            }

    def __init__(self):
        self._default_layer = {}
        super().__init__()
        self.scale = 1
        self.scale_mut = False
        self.extra_param_index_map = ["scale"]
        self.extra_params["scale"] = Parameter(value=self.scale, mutable=self.scale_mut, name="scale")

    def read_from_dol(self, filename):
        try:
            with open(filename, encoding='utf-8') as file:
                try:
                    dol = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                    for row in dol:
                        self.add_layer()
                        self.layer_params[-1]['x'].value = row[1]
                        self.layer_params[-1]['y'].value = row[2]
                        self.layer_params[-1]['z'].value = row[3]
                        self.layer_params[-1]['alpha'].value = row[4]
                        self.layer_params[-1]['beta'].value = row[5]
                        self.layer_params[-1]['gamma'].value = row[6]
                except:  ## Needed for dol files created with PDB units
                    dol = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                    for row in dol:
                        self.add_layer()
                        self.layer_params[-1]['x'].value = row[1]
                        self.layer_params[-1]['y'].value = row[2]
                        self.layer_params[-1]['z'].value = row[3]
                        self.layer_params[-1]['alpha'].value = row[4]
                        self.layer_params[-1]['beta'].value = row[5]
                        self.layer_params[-1]['gamma'].value = row[6]
        except:
            with open(filename, encoding='utf-16') as file:
                dol = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                for row in dol:
                    self.add_layer()
                    self.layer_params[-1]['x'].value = row[1]
                    self.layer_params[-1]['y'].value = row[2]
                    self.layer_params[-1]['z'].value = row[3]
                    self.layer_params[-1]['alpha'].value = row[4]
                    self.layer_params[-1]['beta'].value = row[5]
                    self.layer_params[-1]['gamma'].value = row[6]

    def write_to_dol(self):
        '''For now works only with states built inside the API'''
        if len(self.get_models_by_type('Manual Symmetry')) == 0:
            raise ValueError('Your state has no Manual Symmetries')
        else:
            for ManSym in self.get_models_by_type('Manual Symmetry'):
                if ManSym.name == '':
                    dol_name = '%08d.dol' % (ManSym.model_ptr)
                else:
                    dol_name = ManSym.name + '.dol'

                with open(dol_name, 'w+', encoding='utf-8', newline='') as file:
                    dol = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                    layer_num = 0
                    for layer in ManSym.serialize()['Parameters']:
                        dol.writerow([layer_num, *layer])
                        layer_num += 1

    def get_models_by_type(self, type):
        """
          returns a list of `Models` from the 'Manual Symmetry' field with a given `type_name`.

          :param type: a string of model type , e.g. UniformHollowCylinder.
          :rtype: list of instances of 'Model'
          """
        models = []
        self.get_model_by_type_recursive(self, type, models)
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


for model in hardcode_models:
    ModelFactory.add_model(model)

for model in meta_models:
    ModelFactory.add_model(model)

