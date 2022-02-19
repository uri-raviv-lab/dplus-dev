from dplus.State import State


def test_get_model_by_type():
    state_json = {
        "DomainPreferences": {
            "Convergence": 0.001,
            "DrawDistance": 50,
            "Fitting_UpdateDomain": False,
            "Fitting_UpdateGraph": True,
            "GridSize": 200,
            "LevelOfDetail": 3,
            "OrientationIterations": 100,
            "OrientationMethod": "Monte Carlo (Mersenne Twister)",
            "SignalFile": "",
            "UpdateInterval": 100,
            "UseGrid": False,
            "qMax": 7.5
        },
        "FittingPreferences": {
            "Convergence": 0.1,
            "DerEps": 0.1,
            "DoglegType": "Traditional Dogleg",
            "FittingIterations": 20,
            "LineSearchDirectionType": "",
            "LineSearchType": "Armijo",
            "LossFuncPar1": 0.5,
            "LossFuncPar2": 0.5,
            "LossFunction": "Tolerant Loss",
            "MinimizerType": "Line Search",
            "NonlinearConjugateGradientType": "",
            "StepSize": 0.01,
            "TrustRegionStrategyType": "Levenberg-Marquardt",
            "XRayResidualsType": "Normal Residuals"
        },
        "Viewport": {
            "Axes_at_origin": True,
            "Axes_in_corner": True,
            "Pitch": 35.264385223389,
            "Roll": 0,
            "Yaw": 225.00004577637,
            "Zoom": 8.6602535247803,
            "cPitch": 35.264385223389,
            "cRoll": 0,
            "cpx": -4.9999966621399,
            "cpy": -5.0000033378601,
            "cpz": 4.9999990463257,
            "ctx": 0,
            "cty": 0,
            "ctz": 0
        },
        "Domain": {
            "Geometry": "Domains",
            "ModelPtr": 1,
            "Populations": [
                {
                    "ModelPtr": 2,
                    "Models": [{
                        "Constraints": [[{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                            [{
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }]],
                        "ExtraConstraints": [{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                        "ExtraMutables": [False,
                                          False],
                        "ExtraParameters": [1,
                                            0],
                        "ExtraSigma": [0,
                                       0],
                        "Location": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "LocationConstraints": {
                            "alpha": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "beta": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "gamma": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "x": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "y": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "z": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }
                        },
                        "LocationMutables": {
                            "alpha": False,
                            "beta": False,
                            "gamma": False,
                            "x": False,
                            "y": False,
                            "z": False
                        },
                        "LocationSigma": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "ModelPtr": 4,
                        "Mutables": [[False,
                                      False],
                                     [False,
                                      False]],
                        "Name": "",
                        "Parameters": [[0,
                                        333],
                                       [1,
                                        400]],
                        "Sigma": [[0,
                                   0],
                                  [0,
                                   0]],
                        "Type": ",2",
                        "Use_Grid": True,
                        "nExtraParams": 2,
                        "nLayers": 2,
                        "nlp": 2
                    },
                        {
                        "Constraints": [[{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                            [{
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }]],
                        "ExtraConstraints": [{
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        },
                            {
                            "Link": -1,
                            "MaxIndex": -1,
                            "MaxValue": "inf",
                            "MinIndex": -1,
                            "MinValue": "-inf"
                        }],
                        "ExtraMutables": [False,
                                          False,
                                          False],
                        "ExtraParameters": [1,
                                            0,
                                            10],
                        "ExtraSigma": [0,
                                       0,
                                       0],
                        "Location": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "LocationConstraints": {
                            "alpha": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "beta": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "gamma": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "x": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "y": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            },
                            "z": {
                                "Link": -1,
                                "MaxIndex": -1,
                                "MaxValue": "inf",
                                "MinIndex": -1,
                                "MinValue": "-inf"
                            }
                        },
                        "LocationMutables": {
                            "alpha": False,
                            "beta": False,
                            "gamma": False,
                            "x": False,
                            "y": False,
                            "z": False
                        },
                        "LocationSigma": {
                            "alpha": 0,
                            "beta": 0,
                            "gamma": 0,
                            "x": 0,
                            "y": 0,
                            "z": 0
                        },
                        "ModelPtr": 6,
                        "Mutables": [[False,
                                      False],
                                     [False,
                                      False]],
                        "Name": "",
                        "Parameters": [[0,
                                        333],
                                       [1,
                                        400]],
                        "Sigma": [[0,
                                   0],
                                  [0,
                                   0]],
                        "Type": ",0",
                        "Use_Grid": True,
                        "nExtraParams": 3,
                        "nLayers": 2,
                        "nlp": 2
                    }],
                    "PopulationSize": 1,
                    "PopulationSizeMut": False
                }
            ],
            "Scale": 1,
            "ScaleMut": False
        }
    }
    s = State()
    s.load_from_dictionary(state_json)
    spheres = s.get_models_by_type("Sphere")
    assert len(spheres) == 1


def xtest_get_mutable():
    pass


def test_create_state_with_cylinder():
    from dplus.DataModels.models import UniformHollowCylinder
    uhc = UniformHollowCylinder()
    uhc.name = "test"
    s = State()
    s.Domain.populations[0].add_model(uhc)
    mytest = s.get_model("test")
    assert isinstance(mytest, UniformHollowCylinder)
