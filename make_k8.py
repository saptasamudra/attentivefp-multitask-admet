data = open('attentivefp_moe.py').read()
data = data.replace("'num_experts':         4,", "'num_experts':         8,")
open('attentivefp_moe_k8.py', 'w').write(data)
print('Created attentivefp_moe_k8.py with num_experts=8')
