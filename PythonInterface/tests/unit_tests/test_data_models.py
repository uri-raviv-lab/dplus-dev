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

def test_del_layer():
    from dplus.DataModels.models import ManualSymmetry
    ms = ManualSymmetry()
    for i in range(10):
        ms.add_layer()
        ms.layer_params[i].x.value = i + 0.1
        ms.layer_params[i].y.value = i + 0.2
        ms.layer_params[i].z.value = i + 0.3
    ms.del_layer(0)
    ms.del_layer(range(2, 5))
    ms.del_layer([4, 3])

    assert ((ms.layer_params[2].x.value == 6.1) & (ms.layer_params[2].y.value == 6.2) &
           (ms.layer_params[2].z.value == 6.3)) & (len(ms.layer_params) == 4)
