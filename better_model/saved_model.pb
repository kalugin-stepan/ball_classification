??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.32v2.8.2-130-g92a51d52ad18??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
module_wrapper_1/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*.
shared_namemodule_wrapper_1/dense/kernel
?
1module_wrapper_1/dense/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_1/dense/kernel*
_output_shapes

:d*
dtype0
?
module_wrapper_1/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namemodule_wrapper_1/dense/bias
?
/module_wrapper_1/dense/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_1/dense/bias*
_output_shapes
:*
dtype0
?
module_wrapper_2/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!module_wrapper_2/dense_1/kernel
?
3module_wrapper_2/dense_1/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_2/dense_1/kernel*
_output_shapes

:*
dtype0
?
module_wrapper_2/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namemodule_wrapper_2/dense_1/bias
?
1module_wrapper_2/dense_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_2/dense_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
$Adam/module_wrapper_1/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*5
shared_name&$Adam/module_wrapper_1/dense/kernel/m
?
8Adam/module_wrapper_1/dense/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_1/dense/kernel/m*
_output_shapes

:d*
dtype0
?
"Adam/module_wrapper_1/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/module_wrapper_1/dense/bias/m
?
6Adam/module_wrapper_1/dense/bias/m/Read/ReadVariableOpReadVariableOp"Adam/module_wrapper_1/dense/bias/m*
_output_shapes
:*
dtype0
?
&Adam/module_wrapper_2/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/module_wrapper_2/dense_1/kernel/m
?
:Adam/module_wrapper_2/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_2/dense_1/kernel/m*
_output_shapes

:*
dtype0
?
$Adam/module_wrapper_2/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/module_wrapper_2/dense_1/bias/m
?
8Adam/module_wrapper_2/dense_1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_2/dense_1/bias/m*
_output_shapes
:*
dtype0
?
$Adam/module_wrapper_1/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*5
shared_name&$Adam/module_wrapper_1/dense/kernel/v
?
8Adam/module_wrapper_1/dense/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_1/dense/kernel/v*
_output_shapes

:d*
dtype0
?
"Adam/module_wrapper_1/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/module_wrapper_1/dense/bias/v
?
6Adam/module_wrapper_1/dense/bias/v/Read/ReadVariableOpReadVariableOp"Adam/module_wrapper_1/dense/bias/v*
_output_shapes
:*
dtype0
?
&Adam/module_wrapper_2/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/module_wrapper_2/dense_1/kernel/v
?
:Adam/module_wrapper_2/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_2/dense_1/kernel/v*
_output_shapes

:*
dtype0
?
$Adam/module_wrapper_2/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/module_wrapper_2/dense_1/bias/v
?
8Adam/module_wrapper_2/dense_1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_2/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?
_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
?
_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
_module
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
?
"iter

#beta_1

$beta_2
	%decay
&learning_rate'ml(mm)mn*mo'vp(vq)vr*vs*
 
'0
(1
)2
*3*
 
'0
(1
)2
*3*
* 
?
+layer_regularization_losses
	variables

,layers
-metrics
.layer_metrics
trainable_variables
/non_trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

0serving_default* 
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
* 
* 
* 
?
7layer_regularization_losses
	variables

8layers
9metrics
:layer_metrics
trainable_variables
;non_trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
?

'kernel
(bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses*

'0
(1*

'0
(1*
* 
?
Blayer_regularization_losses
	variables

Clayers
Dmetrics
Elayer_metrics
trainable_variables
Fnon_trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?

)kernel
*bias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*

)0
*1*

)0
*1*
* 
?
Mlayer_regularization_losses
	variables

Nlayers
Ometrics
Player_metrics
trainable_variables
Qnon_trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmodule_wrapper_1/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmodule_wrapper_1/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmodule_wrapper_2/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmodule_wrapper_2/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

R0
S1*
* 
* 
* 
* 
* 
* 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 

'0
(1*

'0
(1*
* 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

)0
*1*

)0
*1*
* 
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
8
	ctotal
	dcount
e	variables
f	keras_api*
H
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

e	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

g0
h1*

j	variables*
?z
VARIABLE_VALUE$Adam/module_wrapper_1/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/module_wrapper_1/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUE&Adam/module_wrapper_2/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE$Adam/module_wrapper_2/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE$Adam/module_wrapper_1/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/module_wrapper_1/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUE&Adam/module_wrapper_2/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE$Adam/module_wrapper_2/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
$serving_default_module_wrapper_inputPlaceholder*/
_output_shapes
:?????????

*
dtype0*$
shape:?????????


?
StatefulPartitionedCallStatefulPartitionedCall$serving_default_module_wrapper_inputmodule_wrapper_1/dense/kernelmodule_wrapper_1/dense/biasmodule_wrapper_2/dense_1/kernelmodule_wrapper_2/dense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_1198
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1module_wrapper_1/dense/kernel/Read/ReadVariableOp/module_wrapper_1/dense/bias/Read/ReadVariableOp3module_wrapper_2/dense_1/kernel/Read/ReadVariableOp1module_wrapper_2/dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp8Adam/module_wrapper_1/dense/kernel/m/Read/ReadVariableOp6Adam/module_wrapper_1/dense/bias/m/Read/ReadVariableOp:Adam/module_wrapper_2/dense_1/kernel/m/Read/ReadVariableOp8Adam/module_wrapper_2/dense_1/bias/m/Read/ReadVariableOp8Adam/module_wrapper_1/dense/kernel/v/Read/ReadVariableOp6Adam/module_wrapper_1/dense/bias/v/Read/ReadVariableOp:Adam/module_wrapper_2/dense_1/kernel/v/Read/ReadVariableOp8Adam/module_wrapper_2/dense_1/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_1387
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratemodule_wrapper_1/dense/kernelmodule_wrapper_1/dense/biasmodule_wrapper_2/dense_1/kernelmodule_wrapper_2/dense_1/biastotalcounttotal_1count_1$Adam/module_wrapper_1/dense/kernel/m"Adam/module_wrapper_1/dense/bias/m&Adam/module_wrapper_2/dense_1/kernel/m$Adam/module_wrapper_2/dense_1/bias/m$Adam/module_wrapper_1/dense/kernel/v"Adam/module_wrapper_1/dense/bias/v&Adam/module_wrapper_2/dense_1/kernel/v$Adam/module_wrapper_2/dense_1/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_1460??
?
?
/__inference_module_wrapper_2_layer_call_fn_1269

args_0
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_932o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
)__inference_sequential_layer_call_fn_1143

inputs
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1057o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
d
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1214

args_0
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   l
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*'
_output_shapes
:?????????d`
IdentityIdentityflatten/Reshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameargs_0
?2
?

__inference__traced_save_1387
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop<
8savev2_module_wrapper_1_dense_kernel_read_readvariableop:
6savev2_module_wrapper_1_dense_bias_read_readvariableop>
:savev2_module_wrapper_2_dense_1_kernel_read_readvariableop<
8savev2_module_wrapper_2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopC
?savev2_adam_module_wrapper_1_dense_kernel_m_read_readvariableopA
=savev2_adam_module_wrapper_1_dense_bias_m_read_readvariableopE
Asavev2_adam_module_wrapper_2_dense_1_kernel_m_read_readvariableopC
?savev2_adam_module_wrapper_2_dense_1_bias_m_read_readvariableopC
?savev2_adam_module_wrapper_1_dense_kernel_v_read_readvariableopA
=savev2_adam_module_wrapper_1_dense_bias_v_read_readvariableopE
Asavev2_adam_module_wrapper_2_dense_1_kernel_v_read_readvariableopC
?savev2_adam_module_wrapper_2_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_module_wrapper_1_dense_kernel_read_readvariableop6savev2_module_wrapper_1_dense_bias_read_readvariableop:savev2_module_wrapper_2_dense_1_kernel_read_readvariableop8savev2_module_wrapper_2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop?savev2_adam_module_wrapper_1_dense_kernel_m_read_readvariableop=savev2_adam_module_wrapper_1_dense_bias_m_read_readvariableopAsavev2_adam_module_wrapper_2_dense_1_kernel_m_read_readvariableop?savev2_adam_module_wrapper_2_dense_1_bias_m_read_readvariableop?savev2_adam_module_wrapper_1_dense_kernel_v_read_readvariableop=savev2_adam_module_wrapper_1_dense_bias_v_read_readvariableopAsavev2_adam_module_wrapper_2_dense_1_kernel_v_read_readvariableop?savev2_adam_module_wrapper_2_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapesx
v: : : : : : :d:::: : : : :d::::d:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 	

_output_shapes
::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1183

inputsG
5module_wrapper_1_dense_matmul_readvariableop_resource:dD
6module_wrapper_1_dense_biasadd_readvariableop_resource:I
7module_wrapper_2_dense_1_matmul_readvariableop_resource:F
8module_wrapper_2_dense_1_biasadd_readvariableop_resource:
identity??-module_wrapper_1/dense/BiasAdd/ReadVariableOp?,module_wrapper_1/dense/MatMul/ReadVariableOp?/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp?.module_wrapper_2/dense_1/MatMul/ReadVariableOpm
module_wrapper/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
module_wrapper/flatten/ReshapeReshapeinputs%module_wrapper/flatten/Const:output:0*
T0*'
_output_shapes
:?????????d?
,module_wrapper_1/dense/MatMul/ReadVariableOpReadVariableOp5module_wrapper_1_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
module_wrapper_1/dense/MatMulMatMul'module_wrapper/flatten/Reshape:output:04module_wrapper_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-module_wrapper_1/dense/BiasAdd/ReadVariableOpReadVariableOp6module_wrapper_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
module_wrapper_1/dense/BiasAddBiasAdd'module_wrapper_1/dense/MatMul:product:05module_wrapper_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
module_wrapper_1/dense/ReluRelu'module_wrapper_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
.module_wrapper_2/dense_1/MatMul/ReadVariableOpReadVariableOp7module_wrapper_2_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
module_wrapper_2/dense_1/MatMulMatMul)module_wrapper_1/dense/Relu:activations:06module_wrapper_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/module_wrapper_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 module_wrapper_2/dense_1/BiasAddBiasAdd)module_wrapper_2/dense_1/MatMul:product:07module_wrapper_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 module_wrapper_2/dense_1/SoftmaxSoftmax)module_wrapper_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????y
IdentityIdentity*module_wrapper_2/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^module_wrapper_1/dense/BiasAdd/ReadVariableOp-^module_wrapper_1/dense/MatMul/ReadVariableOp0^module_wrapper_2/dense_1/BiasAdd/ReadVariableOp/^module_wrapper_2/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 2^
-module_wrapper_1/dense/BiasAdd/ReadVariableOp-module_wrapper_1/dense/BiasAdd/ReadVariableOp2\
,module_wrapper_1/dense/MatMul/ReadVariableOp,module_wrapper_1/dense/MatMul/ReadVariableOp2b
/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp2`
.module_wrapper_2/dense_1/MatMul/ReadVariableOp.module_wrapper_2/dense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
/__inference_module_wrapper_1_layer_call_fn_1229

args_0
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_1_layer_call_and_return_conditional_losses_915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?V
?
 __inference__traced_restore_1460
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: B
0assignvariableop_5_module_wrapper_1_dense_kernel:d<
.assignvariableop_6_module_wrapper_1_dense_bias:D
2assignvariableop_7_module_wrapper_2_dense_1_kernel:>
0assignvariableop_8_module_wrapper_2_dense_1_bias:"
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: J
8assignvariableop_13_adam_module_wrapper_1_dense_kernel_m:dD
6assignvariableop_14_adam_module_wrapper_1_dense_bias_m:L
:assignvariableop_15_adam_module_wrapper_2_dense_1_kernel_m:F
8assignvariableop_16_adam_module_wrapper_2_dense_1_bias_m:J
8assignvariableop_17_adam_module_wrapper_1_dense_kernel_v:dD
6assignvariableop_18_adam_module_wrapper_1_dense_bias_v:L
:assignvariableop_19_adam_module_wrapper_2_dense_1_kernel_v:F
8assignvariableop_20_adam_module_wrapper_2_dense_1_bias_v:
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp0assignvariableop_5_module_wrapper_1_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_module_wrapper_1_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp2assignvariableop_7_module_wrapper_2_dense_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp0assignvariableop_8_module_wrapper_2_dense_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_module_wrapper_1_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_module_wrapper_1_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_adam_module_wrapper_2_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp8assignvariableop_16_adam_module_wrapper_2_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_module_wrapper_1_dense_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_module_wrapper_1_dense_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_adam_module_wrapper_2_dense_1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_module_wrapper_2_dense_1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_1300

args_08
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
C__inference_sequential_layer_call_and_return_conditional_losses_939

inputs&
module_wrapper_1_916:d"
module_wrapper_1_918:&
module_wrapper_2_933:"
module_wrapper_2_935:
identity??(module_wrapper_1/StatefulPartitionedCall?(module_wrapper_2/StatefulPartitionedCall?
module_wrapper/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_module_wrapper_layer_call_and_return_conditional_losses_902?
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall'module_wrapper/PartitionedCall:output:0module_wrapper_1_916module_wrapper_1_918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_1_layer_call_and_return_conditional_losses_915?
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_933module_wrapper_2_935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_932?
IdentityIdentity1module_wrapper_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_1081
module_wrapper_input
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1057o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????


.
_user_specified_namemodule_wrapper_input
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1163

inputsG
5module_wrapper_1_dense_matmul_readvariableop_resource:dD
6module_wrapper_1_dense_biasadd_readvariableop_resource:I
7module_wrapper_2_dense_1_matmul_readvariableop_resource:F
8module_wrapper_2_dense_1_biasadd_readvariableop_resource:
identity??-module_wrapper_1/dense/BiasAdd/ReadVariableOp?,module_wrapper_1/dense/MatMul/ReadVariableOp?/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp?.module_wrapper_2/dense_1/MatMul/ReadVariableOpm
module_wrapper/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
module_wrapper/flatten/ReshapeReshapeinputs%module_wrapper/flatten/Const:output:0*
T0*'
_output_shapes
:?????????d?
,module_wrapper_1/dense/MatMul/ReadVariableOpReadVariableOp5module_wrapper_1_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
module_wrapper_1/dense/MatMulMatMul'module_wrapper/flatten/Reshape:output:04module_wrapper_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-module_wrapper_1/dense/BiasAdd/ReadVariableOpReadVariableOp6module_wrapper_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
module_wrapper_1/dense/BiasAddBiasAdd'module_wrapper_1/dense/MatMul:product:05module_wrapper_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
module_wrapper_1/dense/ReluRelu'module_wrapper_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
.module_wrapper_2/dense_1/MatMul/ReadVariableOpReadVariableOp7module_wrapper_2_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
module_wrapper_2/dense_1/MatMulMatMul)module_wrapper_1/dense/Relu:activations:06module_wrapper_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/module_wrapper_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 module_wrapper_2/dense_1/BiasAddBiasAdd)module_wrapper_2/dense_1/MatMul:product:07module_wrapper_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 module_wrapper_2/dense_1/SoftmaxSoftmax)module_wrapper_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????y
IdentityIdentity*module_wrapper_2/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^module_wrapper_1/dense/BiasAdd/ReadVariableOp-^module_wrapper_1/dense/MatMul/ReadVariableOp0^module_wrapper_2/dense_1/BiasAdd/ReadVariableOp/^module_wrapper_2/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 2^
-module_wrapper_1/dense/BiasAdd/ReadVariableOp-module_wrapper_1/dense/BiasAdd/ReadVariableOp2\
,module_wrapper_1/dense/MatMul/ReadVariableOp,module_wrapper_1/dense/MatMul/ReadVariableOp2b
/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp2`
.module_wrapper_2/dense_1/MatMul/ReadVariableOp.module_wrapper_2/dense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
I__inference_module_wrapper_1_layer_call_and_return_conditional_losses_915

args_06
$dense_matmul_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
c
G__inference_module_wrapper_layer_call_and_return_conditional_losses_902

args_0
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   l
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*'
_output_shapes
:?????????d`
IdentityIdentityflatten/Reshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameargs_0
?
?
"__inference_signature_wrapper_1198
module_wrapper_input
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_889o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????


.
_user_specified_namemodule_wrapper_input
?
?
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_973

args_08
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1260

args_06
$dense_matmul_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1096
module_wrapper_input'
module_wrapper_1_1085:d#
module_wrapper_1_1087:'
module_wrapper_2_1090:#
module_wrapper_2_1092:
identity??(module_wrapper_1/StatefulPartitionedCall?(module_wrapper_2/StatefulPartitionedCall?
module_wrapper/PartitionedCallPartitionedCallmodule_wrapper_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_module_wrapper_layer_call_and_return_conditional_losses_902?
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall'module_wrapper/PartitionedCall:output:0module_wrapper_1_1085module_wrapper_1_1087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_1_layer_call_and_return_conditional_losses_915?
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_1090module_wrapper_2_1092*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_932?
IdentityIdentity1module_wrapper_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall:e a
/
_output_shapes
:?????????


.
_user_specified_namemodule_wrapper_input
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1111
module_wrapper_input'
module_wrapper_1_1100:d#
module_wrapper_1_1102:'
module_wrapper_2_1105:#
module_wrapper_2_1107:
identity??(module_wrapper_1/StatefulPartitionedCall?(module_wrapper_2/StatefulPartitionedCall?
module_wrapper/PartitionedCallPartitionedCallmodule_wrapper_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1024?
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall'module_wrapper/PartitionedCall:output:0module_wrapper_1_1100module_wrapper_1_1102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1003?
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_1105module_wrapper_2_1107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_973?
IdentityIdentity1module_wrapper_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall:e a
/
_output_shapes
:?????????


.
_user_specified_namemodule_wrapper_input
?
I
-__inference_module_wrapper_layer_call_fn_1208

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1024`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameargs_0
?
?
)__inference_sequential_layer_call_fn_1130

inputs
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
d
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1024

args_0
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   l
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*'
_output_shapes
:?????????d`
IdentityIdentityflatten/Reshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameargs_0
?
I
-__inference_module_wrapper_layer_call_fn_1203

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_module_wrapper_layer_call_and_return_conditional_losses_902`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameargs_0
?
?
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_1289

args_08
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
(__inference_sequential_layer_call_fn_950
module_wrapper_input
unknown:d
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????


.
_user_specified_namemodule_wrapper_input
?
d
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1220

args_0
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   l
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*'
_output_shapes
:?????????d`
IdentityIdentityflatten/Reshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

:W S
/
_output_shapes
:?????????


 
_user_specified_nameargs_0
?
?
/__inference_module_wrapper_1_layer_call_fn_1238

args_0
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
?
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1003

args_06
$dense_matmul_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
?
__inference__wrapped_model_889
module_wrapper_inputR
@sequential_module_wrapper_1_dense_matmul_readvariableop_resource:dO
Asequential_module_wrapper_1_dense_biasadd_readvariableop_resource:T
Bsequential_module_wrapper_2_dense_1_matmul_readvariableop_resource:Q
Csequential_module_wrapper_2_dense_1_biasadd_readvariableop_resource:
identity??8sequential/module_wrapper_1/dense/BiasAdd/ReadVariableOp?7sequential/module_wrapper_1/dense/MatMul/ReadVariableOp?:sequential/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp?9sequential/module_wrapper_2/dense_1/MatMul/ReadVariableOpx
'sequential/module_wrapper/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
)sequential/module_wrapper/flatten/ReshapeReshapemodule_wrapper_input0sequential/module_wrapper/flatten/Const:output:0*
T0*'
_output_shapes
:?????????d?
7sequential/module_wrapper_1/dense/MatMul/ReadVariableOpReadVariableOp@sequential_module_wrapper_1_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
(sequential/module_wrapper_1/dense/MatMulMatMul2sequential/module_wrapper/flatten/Reshape:output:0?sequential/module_wrapper_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
8sequential/module_wrapper_1/dense/BiasAdd/ReadVariableOpReadVariableOpAsequential_module_wrapper_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)sequential/module_wrapper_1/dense/BiasAddBiasAdd2sequential/module_wrapper_1/dense/MatMul:product:0@sequential/module_wrapper_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&sequential/module_wrapper_1/dense/ReluRelu2sequential/module_wrapper_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
9sequential/module_wrapper_2/dense_1/MatMul/ReadVariableOpReadVariableOpBsequential_module_wrapper_2_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
*sequential/module_wrapper_2/dense_1/MatMulMatMul4sequential/module_wrapper_1/dense/Relu:activations:0Asequential/module_wrapper_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
:sequential/module_wrapper_2/dense_1/BiasAdd/ReadVariableOpReadVariableOpCsequential_module_wrapper_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
+sequential/module_wrapper_2/dense_1/BiasAddBiasAdd4sequential/module_wrapper_2/dense_1/MatMul:product:0Bsequential/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential/module_wrapper_2/dense_1/SoftmaxSoftmax4sequential/module_wrapper_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity5sequential/module_wrapper_2/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp9^sequential/module_wrapper_1/dense/BiasAdd/ReadVariableOp8^sequential/module_wrapper_1/dense/MatMul/ReadVariableOp;^sequential/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp:^sequential/module_wrapper_2/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 2t
8sequential/module_wrapper_1/dense/BiasAdd/ReadVariableOp8sequential/module_wrapper_1/dense/BiasAdd/ReadVariableOp2r
7sequential/module_wrapper_1/dense/MatMul/ReadVariableOp7sequential/module_wrapper_1/dense/MatMul/ReadVariableOp2x
:sequential/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp:sequential/module_wrapper_2/dense_1/BiasAdd/ReadVariableOp2v
9sequential/module_wrapper_2/dense_1/MatMul/ReadVariableOp9sequential/module_wrapper_2/dense_1/MatMul/ReadVariableOp:e a
/
_output_shapes
:?????????


.
_user_specified_namemodule_wrapper_input
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1057

inputs'
module_wrapper_1_1046:d#
module_wrapper_1_1048:'
module_wrapper_2_1051:#
module_wrapper_2_1053:
identity??(module_wrapper_1/StatefulPartitionedCall?(module_wrapper_2/StatefulPartitionedCall?
module_wrapper/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1024?
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall'module_wrapper/PartitionedCall:output:0module_wrapper_1_1046module_wrapper_1_1048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1003?
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_1051module_wrapper_2_1053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_973?
IdentityIdentity1module_wrapper_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????

: : : : 2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1249

args_06
$dense_matmul_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
?
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_932

args_08
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
/__inference_module_wrapper_2_layer_call_fn_1278

args_0
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_2_layer_call_and_return_conditional_losses_973o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
]
module_wrapper_inputE
&serving_default_module_wrapper_input:0?????????

D
module_wrapper_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?v
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?
_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
_module
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"iter

#beta_1

$beta_2
	%decay
&learning_rate'ml(mm)mn*mo'vp(vq)vr*vs"
tf_deprecated_optimizer
<
'0
(1
)2
*3"
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+layer_regularization_losses
	variables

,layers
-metrics
.layer_metrics
trainable_variables
/non_trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_sequential_layer_call_fn_950
)__inference_sequential_layer_call_fn_1130
)__inference_sequential_layer_call_fn_1143
)__inference_sequential_layer_call_fn_1081?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_1163
D__inference_sequential_layer_call_and_return_conditional_losses_1183
D__inference_sequential_layer_call_and_return_conditional_losses_1096
D__inference_sequential_layer_call_and_return_conditional_losses_1111?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_889?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *;?8
6?3
module_wrapper_input?????????


,
0serving_default"
signature_map
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7layer_regularization_losses
	variables

8layers
9metrics
:layer_metrics
trainable_variables
;non_trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_module_wrapper_layer_call_fn_1203
-__inference_module_wrapper_layer_call_fn_1208?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1214
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1220?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?

'kernel
(bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Blayer_regularization_losses
	variables

Clayers
Dmetrics
Elayer_metrics
trainable_variables
Fnon_trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_module_wrapper_1_layer_call_fn_1229
/__inference_module_wrapper_1_layer_call_fn_1238?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1249
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1260?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?

)kernel
*bias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mlayer_regularization_losses
	variables

Nlayers
Ometrics
Player_metrics
trainable_variables
Qnon_trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_module_wrapper_2_layer_call_fn_1269
/__inference_module_wrapper_2_layer_call_fn_1278?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_1289
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_1300?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-d2module_wrapper_1/dense/kernel
):'2module_wrapper_1/dense/bias
1:/2module_wrapper_2/dense_1/kernel
+:)2module_wrapper_2/dense_1/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?B?
"__inference_signature_wrapper_1198module_wrapper_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
N
	ctotal
	dcount
e	variables
f	keras_api"
_tf_keras_metric
^
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
c0
d1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
4:2d2$Adam/module_wrapper_1/dense/kernel/m
.:,2"Adam/module_wrapper_1/dense/bias/m
6:42&Adam/module_wrapper_2/dense_1/kernel/m
0:.2$Adam/module_wrapper_2/dense_1/bias/m
4:2d2$Adam/module_wrapper_1/dense/kernel/v
.:,2"Adam/module_wrapper_1/dense/bias/v
6:42&Adam/module_wrapper_2/dense_1/kernel/v
0:.2$Adam/module_wrapper_2/dense_1/bias/v?
__inference__wrapped_model_889?'()*E?B
;?8
6?3
module_wrapper_input?????????


? "C?@
>
module_wrapper_2*?'
module_wrapper_2??????????
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1249l'(??<
%?"
 ?
args_0?????????d
?

trainingp "%?"
?
0?????????
? ?
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_1260l'(??<
%?"
 ?
args_0?????????d
?

trainingp"%?"
?
0?????????
? ?
/__inference_module_wrapper_1_layer_call_fn_1229_'(??<
%?"
 ?
args_0?????????d
?

trainingp "???????????
/__inference_module_wrapper_1_layer_call_fn_1238_'(??<
%?"
 ?
args_0?????????d
?

trainingp"???????????
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_1289l)*??<
%?"
 ?
args_0?????????
?

trainingp "%?"
?
0?????????
? ?
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_1300l)*??<
%?"
 ?
args_0?????????
?

trainingp"%?"
?
0?????????
? ?
/__inference_module_wrapper_2_layer_call_fn_1269_)*??<
%?"
 ?
args_0?????????
?

trainingp "???????????
/__inference_module_wrapper_2_layer_call_fn_1278_)*??<
%?"
 ?
args_0?????????
?

trainingp"???????????
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1214pG?D
-?*
(?%
args_0?????????


?

trainingp "%?"
?
0?????????d
? ?
H__inference_module_wrapper_layer_call_and_return_conditional_losses_1220pG?D
-?*
(?%
args_0?????????


?

trainingp"%?"
?
0?????????d
? ?
-__inference_module_wrapper_layer_call_fn_1203cG?D
-?*
(?%
args_0?????????


?

trainingp "??????????d?
-__inference_module_wrapper_layer_call_fn_1208cG?D
-?*
(?%
args_0?????????


?

trainingp"??????????d?
D__inference_sequential_layer_call_and_return_conditional_losses_1096|'()*M?J
C?@
6?3
module_wrapper_input?????????


p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1111|'()*M?J
C?@
6?3
module_wrapper_input?????????


p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1163n'()*??<
5?2
(?%
inputs?????????


p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1183n'()*??<
5?2
(?%
inputs?????????


p

 
? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_1081o'()*M?J
C?@
6?3
module_wrapper_input?????????


p

 
? "???????????
)__inference_sequential_layer_call_fn_1130a'()*??<
5?2
(?%
inputs?????????


p 

 
? "???????????
)__inference_sequential_layer_call_fn_1143a'()*??<
5?2
(?%
inputs?????????


p

 
? "???????????
(__inference_sequential_layer_call_fn_950o'()*M?J
C?@
6?3
module_wrapper_input?????????


p 

 
? "???????????
"__inference_signature_wrapper_1198?'()*]?Z
? 
S?P
N
module_wrapper_input6?3
module_wrapper_input?????????

"C?@
>
module_wrapper_2*?'
module_wrapper_2?????????