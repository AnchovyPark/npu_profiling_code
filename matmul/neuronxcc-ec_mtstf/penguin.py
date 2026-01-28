import neuronxcc.starfish.penguin.ir.ir as m0
import neuronxcc.starfish.penguin.ir.DebugInfo as m1
import neuronxcc.starfish.penguin.targets.tonga.APIndex as m2
import neuronxcc.starfish.penguin.targets.tonga.TongaInst as m3
import neuronxcc.starfish.penguin.targets.tonga.TongaISAInst as m4
import neuronxcc.starfish.penguin.targets.tonga.TongaTensor as m5
import numpy as np
v0 = m0.Function(id_=0, batch_ids=[], attrs=("model-type=memory-bound","mac-count=1160",'hlo-metrics={"AliasedOutputSize":0,"ArithmeticIntensity":0.99656355381011963,"ConstantSize":0,"HloInputCount":-1,"HloMacCount":1160,"HloOutputCount":-1,"IfmapSize":0,"OfmapSize":0,"OutputsReadFromCount":-1,"PassthroughTensorsCount":-1,"RedundantOutputCount":-1,"Traffic":2328}'))
def weight_load(p):
  t = np.load(p)
  return t
import neuronxcc.starfish.support as m7
v1 = m0.Tensor(name="input0", shape=(2,2), parent=v0, id=1, dtype="float16", view=m0.TensorView(shape=(2,2), layout="NC", transpose=(0,1)), attrs={'CrossPassTensor': ""})
v0.markInput(v1)
v2 = m0.Tensor(name="input1", shape=(2,290), parent=v0, id=2, dtype="float16", view=m0.TensorView(shape=(2,290), layout="NC", transpose=(0,1)), attrs={'CrossPassTensor': ""})
v0.markInput(v2)
v4 = m0.Tensor(name="output0", shape=(2,290), parent=v0, id=3, dtype="float16", view=m0.TensorView(shape=(2,290), layout="NC", transpose=(0,1)), )
import neuronxcc.starfish.penguin.frontends.XlaFE as m8
v3 = m8.NeuronTensorOp(srcs=[v1, v2], dsts=[v4], xla_op='mhlo.dot', lhs_batching_dims=[], lhs_contract_dims=[1], rhs_batching_dims=[], rhs_contract_dims=[0], id=4, parent=v0, dl=m1.DebugLocation(tensor_op_name="_dot.3", file="", line=0, column=0, hlo_id=3))
v0.markOutput(v4)
v0.id=5
ir=v0
