import torch
import os
from Modules import read_data, train_model, optimize
from Modules import RuleSet, RuleBasedTPP

# Parameters setting
file_path = "data/demo.csv"     # data file path
target_name = "Y"               # target variable
data_name = "demo"              # data name
max_order = 2                   # maximum order of rules
num_candidates = 10             # number of candidate rules
n_calls = 20                    # number of optimization calls
device = "cpu"                  # device "cpu" or "cuda"
lr = 0.01                       # learning rate
num_epochs = 100                # number of epochs
patience = 5                    # patience for early stopping

# Train data and optimize rules
data, var_name_dict = read_data(file_path, target_varibles=target_name, outliers=0.0)
print(f"The data have {len(data)} samples.")
best_rules, best_loss = optimize(
    data, var_name_dict, target_name, max_order=max_order, 
    num_candidates=num_candidates, n_calls=n_calls, device=device
)
print("Best Rules:", best_rules)
print("Training model using the best rules...")
rule_set = RuleSet(data, var_name_dict)
rules = best_rules
for rule in rules:
    rule_set.add_rule(rule)
model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
model.to(device)
loss, output = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs, lr, patience, if_print=False)
print("Saving results to file...")
os.makedirs('results', exist_ok=True)
with open(f'results/results_{data_name}.txt', 'w') as f:
    count = 0
    for rule_name in model.rule_name_dict:
        weight = round(torch.exp(model.rule_weights[model.rule_name_dict[rule_name]]).item(), 4)
        count += 1
        f.write(f"| {count:2} | {rule_name} -> {target_name} \t| weight={weight:.4f} |\n")
    loss = output[-1]
    f.write(f"Train Loss: {loss[0]:.4f},    Test NLL: {loss[1]:.4f},    Test RMSE: {loss[2]:.4f}")
print("Finished!")
