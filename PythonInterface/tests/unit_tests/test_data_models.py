from dplus.DataModels import Parameter
from dplus.State import State

def test_create_state_with_cylinder():
    from dplus.DataModels.models import UniformHollowCylinder
    uhc=UniformHollowCylinder()
    uhc.name="test_hc"



def test_add_layer():
    from dplus.DataModels.models import ManualSymmetry, UniformHollowCylinder
    ms=ManualSymmetry()
    ms.children.append(UniformHollowCylinder())
    ms.add_layer()
    for param in ms.layer_params[0]:
        ms.layer_params[0][param].value=1
