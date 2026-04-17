import torch
import numpy as np
import sympy as sp
import os, sys
import symbolicregression
from symbolicregression.model.levin_tree import LevinTree
import requests
from IPython.display import display
import traceback

torch.manual_seed(0)
np.random.seed(0)

model_path = "model.pt" 
try:
    if not os.path.isfile(model_path): 
        url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    if not torch.cuda.is_available():
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
        model = model.cuda()
    print(model.device)
    print("Model successfully loaded!")

except Exception as e:
    print("ERROR: model not loaded! path was: {}".format(model_path))
    print(e)
    traceback.print_exc()
    exit()

est = symbolicregression.model.SymbolicTransformerRegressor(
                        model=model,
                        max_input_points=200,
                        n_trees_to_refine=100,
                        rescale=True
                        )
#print(est.model.decoder.word2id)
#exit()

##Example of data

x = np.random.randn(100, 2)
y = np.cos(2*np.pi*x[:,0])+x[:,1]**2

print("\n=================================================== Original ===================================================================")
torch.manual_seed(114514)
np.random.seed(1919810)

est.fit(x,y)

#actions_taken = est.model.decoder.actions_taken
#print(actions_taken)
replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
for op,replace_op in replace_ops.items():
    model_str = model_str.replace(op,replace_op)
display(sp.parse_expr(model_str))

#print("\n====================================================== Debugging ================================================================")
#torch.manual_seed(114514)
#np.random.seed(1919810)
#state = est.init_state(x,y)
#i = 0
#while est.is_solution(state) == False:
#    #a = actions_taken[i]  # This will not crash so long as we have is_solution correct, otherwise we may 
#                          #   index out of bound
#    policy = est.get_policy(state)
#    a = policy.argmax()
#    print(est.model.decoder.id2word[a], end=" ")
#    state = est.apply_action(state, idx=a)
#    i = i + 1
#print()
#assert(i == len(actions_taken)) # this checks that we applied all actions as the original model
#assert(est.is_solution(state) == True)
#est.prep_for_refinement(state)
#model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
#replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
#for op,replace_op in replace_ops.items():
#    model_str = model_str.replace(op,replace_op)
#display(sp.parse_expr(model_str))

print("\n=================================================== Levin Tree Search ===================================================================")
torch.manual_seed(114514)
np.random.seed(1919810)

state = est.init_state(x,y)
lts = LevinTree(est.model, state)
final_state = lts.search()
est.prep_for_refinement(final_state)

model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
for op,replace_op in replace_ops.items():
    model_str = model_str.replace(op,replace_op)
display(sp.parse_expr(model_str))
