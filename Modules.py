import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import random
from tqdm import tqdm
from itertools import product
import time
from skopt import forest_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args
from itertools import product


# Data Loading
def read_data(file_path, scale=True, outliers=0.0, target_varibles=None):
    """
    Reads a CSV file, processes the data by handling outliers, normalizing values, and mapping variable names to integers.

    Parameters:
    file_path (str): Path to the CSV file containing the data.
    scale (bool, default=True): If True, normalizes the 'v' values within each 'k' group.
    outliers (float, default=0.0): The fraction of data to clip at the upper and lower quantiles (0-1).
    target_varibles (str or None, default=None): If provided, this variable name is moved to the first position in the list of variables.
    Returns:
    data (list of lists): Nested list where each inner list corresponds to a group identified by 'id'.
                          Each row in a group contains [time ('t'), variable index ('k'), value ('v')].
    var_name_dict (dict): A dictionary mapping the variable names to integer indices.
    """
    df = pd.read_csv(file_path, header=0, sep=",")
    var_name_list = sorted(set(df['k'].unique()))
    if target_varibles:
        if target_varibles in var_name_list:
            var_name_list.remove(target_varibles)
            var_name_list.insert(0, target_varibles)
    var_name_dict = {val: idx for idx, val in enumerate(var_name_list)}
    unique_k = df['k'].unique()
    for k_value in unique_k:
        subset = df[df['k'] == k_value]
        if k_value.endswith('High'):
            q_upper = subset['v'].quantile(1-outliers)
            df.loc[subset.index, 'v'] = subset['v'].clip(upper=q_upper)
        elif k_value.endswith('Low'):
            q_lower = subset['v'].quantile(outliers)
            df.loc[subset.index, 'v'] = subset['v'].clip(lower=q_lower)
    if scale:
        df['v_normalized'] = df.groupby('k')['v'].transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.0)
        df['v'] = df['v_normalized']
        df = df.drop(columns=['v_normalized'])
    df['k'] = df['k'].map(var_name_dict)
    df = df.sort_values(by=['id', 't', 'k']).reset_index(drop=True)
    data = [
        [[round(row.t,6), int(row.k), round(row.v,6)] for _, row in group.iterrows()]
        for _, group in df.groupby('id')
    ]
    return data, var_name_dict


class RuleSet():
    def __init__(self, event_data, var_name_dict):
        """
        Initializes the RuleSet object with the provided event data and variable name dictionary.

        Parameters:
        event_data (list): A list of events, where each event contains a time, variable index, and value.
        var_name_dict (dict): A dictionary that maps variable names to unique identifiers.
        """
        self.event_data = event_data
        self.var_name_dict = var_name_dict
        self.rule_name_dict = dict()
        self.rule_name_set = set()
        self.rule_var_ids = set()
        self.var_count = 0
        self.rule_event_data = [[] for _ in range(len(event_data))]
        self.time_tolerance = 0.1
    
    def add_rule(self, cause):
        """
        Adds a rule to the RuleSet, processing its cause expression into RPN and updating event data.

        Parameters:
        cause (str): The rule's cause expression, which can be a logical combination of variables.
        """
        cause_rpn = self.infix_to_rpn(self.replace_variables(cause))
        cause_var_ids = set([int(token) for token in cause_rpn if token.isdigit()])
        cause_name = self.rpn_to_rule(cause_rpn)
        if cause_name not in self.rule_name_dict:
            self.rule_name_dict[cause_name] = self.var_count
            self.rule_var_ids.update(cause_var_ids)
            cause_rpn_str = " ".join(cause_rpn)
            self.rule_name_set.add(cause_rpn_str)
            for ind, events in enumerate(self.event_data):
                cause_value, cause_time = self.get_new_var(events, cause_rpn)
                self.rule_event_data[ind].extend([t, self.var_count, v] for t, v in zip(cause_time, cause_value))
            self.var_count += 1
            
    def replace_variables(self, rule_expression):
        """
        Replaces variable names in the rule expression with their corresponding numeric IDs.

        Parameters:
        rule_expression (str): The rule expression containing variable names.
        Returns:
        str: The rule expression with variable names replaced by IDs.
        """
        var_name_dict = self.var_name_dict
        def replace_var(match):
            var_name = match.group(0)
            return str(var_name_dict.get(var_name, var_name)) 
        pattern = r'\b(' + '|'.join(re.escape(key) for key in var_name_dict.keys()) + r')\b'
        result = re.sub(pattern, replace_var, rule_expression)
        return result
    
    def infix_to_rpn(self, expression):
        """
        Converts an infix expression (e.g., "A equal B") into Reverse Polish Notation (RPN).

        Parameters:
        expression (str): The infix expression to convert.
        Returns:
        list: A list representing the RPN form of the expression.
        """
        precedence = {'equal': 1, 'before': 1, 'and': 1}
        output = []
        operators = []
        tokens = expression.replace('(', ' ( ').replace(')', ' ) ').split()
        last_token = None
        open_parentheses = 0
        for token in tokens:
            if token.lower() in precedence:
                if last_token is None or last_token.lower() in precedence or last_token == '(':
                    raise ValueError(f"Invalid expression: operator {token} in wrong position.")
                while (operators and operators[-1] in precedence and
                    precedence[operators[-1]] >= precedence[token.lower()]):
                    output.append(operators.pop())
                operators.append(token.lower())
            elif token == '(':
                open_parentheses += 1
                operators.append(token)
            elif token == ')':
                open_parentheses -= 1
                if open_parentheses < 0:
                    raise ValueError("Invalid expression: mismatched parentheses.")
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()
            elif token.isdigit():
                if last_token and last_token.isdigit():
                    raise ValueError("Invalid expression: two consecutive operands.")
                output.append(token)
            else:
                raise ValueError(f'Invalid token {token} in expression.') 
            last_token = token
        if open_parentheses != 0:
            raise ValueError("Invalid expression: mismatched parentheses.")
        if last_token and last_token.lower() in precedence:
            raise ValueError("Invalid expression: cannot end with an operator.")
        while operators:
            op = operators.pop()
            if op == '(' or op == ')':
                raise ValueError("Invalid expression: mismatched parentheses.")
            output.append(op)
        return output
    
    def rpn_to_rule(self, rpn_expression):
        """
        Converts a Reverse Polish Notation (RPN) expression into a human-readable rule string.

        Parameters:
        rpn_expression (list): The RPN expression to convert.
        Returns:
        str: A human-readable rule expression.
        """
        reverse_var_dict = {str(v): k for k, v in self.var_name_dict.items()}
        stack = []
        for token in rpn_expression:
            if token.isdigit():
                var_name = reverse_var_dict[token]
                stack.append(var_name)
            else:
                if token.lower() in ['before', 'equal', 'and']:
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    sub_expression = f"({operand1} {token} {operand2})"
                    stack.append(sub_expression)
                else:
                    raise ValueError(f"Invalid token {token} in RPN expression.")
        final_expression = stack.pop()
        if final_expression[0] == "(" and final_expression[-1] == ")":
            return final_expression[1:-1]
        return final_expression
    
    def get_new_var(self, events, rpn_expression):
        """
        Evaluates the RPN expression for each event in the dataset and returns the result.

        Parameters:
        events (list): A list of events, each containing a time, variable index, and value.
        rpn_expression (list): The RPN expression to evaluate.
        Returns:
        tuple: A tuple containing the evaluated variable values and corresponding times.
        """
        events_dict = {}
        for var_name_id in self.var_name_dict.values():
            events_dict[str(var_name_id)] = {
                'time': [],
                'value': []
            }
        for t_j, k_j, v_j in events:
            events_dict[str(int(k_j))]['time'].append(t_j)
            events_dict[str(int(k_j))]['value'].append(1)
        def get_variable_info(variables, var_id):
            return variables[var_id]['value'], variables[var_id]['time']
        def before(var1, var2):
            value1, time1 = var1
            value2, time2 = var2
            result_value = []
            result_time = []
            i, j = 0, 0
            while i < len(time1) and j < len(time2):
                if time1[i] < time2[j] - self.time_tolerance:
                    result_value.append(1)
                    result_time.append(max(time1[i], time2[j]))
                    i += 1
                    j += 1
                else:
                    j += 1
            return result_value, result_time
        def equal(var1, var2):
            value1, time1 = var1
            value2, time2 = var2
            result_value = []
            result_time = []
            i, j = 0, 0
            while i < len(time1) and j < len(time2):
                if abs(time1[i] - time2[j]) <= self.time_tolerance:
                    if value1[i] * value2[j] != 0:
                        result_value.append(1)
                        result_time.append(max(time1[i], time2[j]))
                    i += 1
                    j += 1
                elif time1[i] < time2[j]:
                    i += 1
                else:
                    j += 1
            return result_value, result_time
        def and_op(var1, var2): 
            value1, time1 = var1
            value2, time2 = var2
            result_value = []
            result_time = []
            i, j = 0, 0
            while i < len(time1) and j < len(time2):
                if abs(time1[i] - time2[j]) > self.time_tolerance:
                    result_value.append(1)
                    result_time.append(max(time1[i], time2[j]))
                    i += 1
                    j += 1
                else:
                    if i+1 < len(time1) and j+1 < len(time2):
                        if time1[i+1] < time2[j+1]:
                            i += 1
                        else:
                            j += 1
                    elif i+1 < len(time1):
                        i += 1
                    else:
                        j += 1
            return result_value, result_time
        stack = []
        for token in rpn_expression:
            if token == 'before':
                var2 = stack.pop()
                var1 = stack.pop()
                stack.append(before(var1, var2))
            elif token == 'equal':
                var2 = stack.pop()
                var1 = stack.pop()
                stack.append(equal(var1, var2))
            elif token == 'and':
                var2 = stack.pop()
                var1 = stack.pop()
                stack.append(and_op(var1, var2))
            else:
                stack.append(get_variable_info(events_dict, token))
        result_value, result_time = stack.pop()
        return result_value, result_time
    

class RuleBasedTPP(nn.Module):
    def __init__(self, var_name_dict, rule_name_dict, rule_var_ids, device="cpu"):
        """
        Initializes the Rule-Based Temporal Point Process (RTPP) model.
        
        Parameters:
        - var_name_dict (dict): Dictionary mapping variable names to unique identifiers.
        - rule_name_dict (dict): Dictionary mapping rule names to unique identifiers.
        - rule_var_ids (set): Set of variable IDs used in rule expressions.
        - device (str): The device (e.g., 'cpu' or 'cuda') where the model will be executed.
        """
        super(RuleBasedTPP, self).__init__()
        self.var_name_dict = var_name_dict # Dictionary mapping variable names to unique identifiers
        self.rule_name_dict = rule_name_dict # Dictionary mapping rule names to unique identifiers
        self.rule_var_ids = rule_var_ids # Set of variable IDs used in rule expressions
        self.device = torch.device(device) # Device for computation
        self.K = len(var_name_dict) # Number of event types
        self.M = len(rule_name_dict) # Number of rule events
        self.mu = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))                # Base intensity parameter μ
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=self.device))              # Softplus parameter β
        self.rule_weights = nn.Parameter(torch.rand(self.M, dtype=torch.float32, device=self.device))     # Rule-guided strength for cause event type m
        self.numf_weights = nn.Parameter(torch.rand(self.K, dtype=torch.float32, device=self.device))     # Measurement-driven strength for each event type k_x
        self.numf_weights_mask = torch.zeros(self.K, dtype=torch.float32, device=self.device, requires_grad=False)  # Mask for measurement weights
        for var_id in self.rule_var_ids:
            self.numf_weights_mask[var_id] = 1.0
        self.numf_weights_mask[0] = 1.0
        
    def forward(self, event_times, event_types, event_meass, rule_times, rule_types, rule_meass):
        """
        Performs the forward pass of the model, computing the loss (negative log-likelihood, time, and type losses).
        
        Parameters:
        - event_times (Tensor): The times of events.
        - event_types (Tensor): The types of events.
        - event_meass (Tensor): The measurements associated with events.
        - rule_times (Tensor): The times of rule events.
        - rule_types (Tensor): The types of rule events.
        - rule_meass (Tensor): The measurements associated with rule events.
        Returns:
        - nll + type_loss + time_loss (Tensor): The total loss function (negative log-likelihood, type loss, and time loss).
        """
        # Negative Log-Likelihood (NLL) loss
        given_times = event_times[event_types == 0]
        lambda_values = self.intensity(given_times=given_times, 
                                       event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                       rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        log_likelihood = torch.sum(torch.log(lambda_values))
        T_max = torch.max(given_times)
        T_min = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        num_samples = 20
        t_values = torch.linspace(T_min, T_max, num_samples, device=self.device)
        integral_values = self.intensity(given_times=t_values, 
                                         event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                         rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        integral = torch.trapz(integral_values, t_values)
        nll = - (log_likelihood - integral)
        return nll

    def intensity(self, given_times, event_times, event_types, event_meass, rule_times, rule_types, rule_meass):
        """
        Computes the intensity function (λ) for each event type at the given times, considering the history of events.

        Parameters:
        - given_times (Tensor): The times at which the intensity is evaluated.
        - event_times (Tensor): The times of historical events.
        - event_types (Tensor): The types of historical events.
        - event_meass (Tensor): The measurements associated with historical events.
        - rule_times (Tensor): The times of rule events.
        - rule_types (Tensor): The types of rule events.
        - rule_meass (Tensor): The measurements associated with rule events.
        Returns:
        - total_intensity (Tensor): The computed intensity for each event type at the given times.
        """
        # Base intensity component
        base_intensity = torch.tensor(0.0, dtype=torch.float32, device=self.device) #self.mu
        # Rule-guided intensity component
        rule_intensity = torch.sum(rule_meass * self.time_decay(given_times.view(-1,1)-rule_times) * self.rule_weights[rule_types], dim=1)
        # Measurement-driven intensity component
        numf_intensity = torch.sum(event_meass * self.time_decay(given_times.view(-1,1)-event_times) * self.numf_weights[event_types] * self.numf_weights_mask[event_types], dim=1)
        # Total intensity
        #base_intensity = base_intensity.expand_as(given_times)
        sum_intensity = base_intensity + rule_intensity + numf_intensity 
        # Softplus function
        total_intensity = torch.log1p(torch.exp(self.beta * sum_intensity)) / self.beta
        return total_intensity

    def time_decay(self, delta_t):
        """
        Computes the time decay for a given time difference (delta_t) using an exponential decay function.

        Parameters:
        - delta_t (Tensor): The time differences between events.
        Returns:
        - decay (Tensor): The decay values based on the time differences.
        """
        decay = torch.exp(-delta_t)
        decay[delta_t <= 0] = 0.0
        return decay

    def evaluate(self, event_times, event_types, event_meass, rule_times, rule_types, rule_meass, target_name):
        """
        Evaluates the model performance using Negative Log-Likelihood (NLL), Mean Absolute Error (MAE), and Root Mean Square Error (RMSE).
        
        Parameters:
        - event_times (Tensor): The times of events.
        - event_types (Tensor): The types of events.
        - event_meass (Tensor): The measurements associated with events.
        - rule_times (Tensor): The times of rule events.
        - rule_types (Tensor): The types of rule events.
        - rule_meass (Tensor): The measurements associated with rule events.
        - target_name (str): The name of the target variable for evaluation.
        Returns:
        - nll_k (Tensor): The negative log-likelihood for the target variable.
        - mae (Tensor): The mean absolute error of the predicted intervals.
        - rmse (Tensor): The root mean square error of the predicted intervals.
        """
        target_id = self.var_name_dict[target_name]
        target_indices = (event_types == target_id).nonzero(as_tuple=True)[0]
        lambda_values = self.intensity(given_times=event_times[target_indices], 
                                       event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                       rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        log_likelihood = torch.sum(torch.log(lambda_values))
        T_max = torch.max(event_times[target_indices])
        T_min = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        num_samples = 20
        t_values = torch.linspace(T_min, T_max, num_samples, device=self.device)
        integral_values = self.intensity(given_times=t_values, 
                                         event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                         rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
        integral = torch.trapz(integral_values, t_values)
        nll_k = - (log_likelihood - integral) # Negative Log-Likelihood
        target_indices = target_indices[1:] if target_indices[0] == 0 else target_indices
        if len(target_indices) == 0:
            return nll_k, torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0.0, dtype=torch.float32, device=self.device)
        prev_indices = target_indices - 1
        target_times = event_times[target_indices]
        prev_times = event_times[prev_indices]
        time_intervals = torch.tensor([self.MC_next_event(prev_time, event_times, event_types, event_meass, rule_times, rule_types, rule_meass) for prev_time in prev_times], device=self.device)
        real_intervals = target_times - prev_times
        interval_diffs = time_intervals - real_intervals
        mae = torch.mean(torch.abs(interval_diffs)) # Mean Absolute Error
        mse = torch.mean(torch.square(interval_diffs))
        rmse = mse.sqrt() # Root Mean Square Error
        return nll_k, mae, rmse
    
    def MC_next_event(self, prev_times, event_times, event_types, event_meass, rule_times, rule_types, rule_meass, max_time=6.0, num_samples=2000):
        """
        Predicts the time of the next event using the Monte Carlo method.
        
        Parameters:
        - prev_times (float): The time of the previous event.
        - event_times (list): List of times when historical events occurred.
        - event_types (list): List of types of historical events.
        - event_meass (list): List of measurements associated with historical events.
        - rule_times (list): List of times when rule events occurred.
        - rule_types (list): List of types of rule events.
        - rule_meass (list): List of measurements associated with rule events.
        - max_time (float): The maximum time range for prediction from the previous event.
        - num_samples (int): The number of samples to generate.
        Returns:
        - torch.Tensor: The average predicted time of the next event.
        """
        next_times = []
        for _ in range(num_samples):
            t_current = prev_times.clone()
            max_iterations = 100
            iteration = 0
            while iteration < max_iterations:
                current_intensity = self.intensity(given_times=torch.tensor([t_current], dtype=torch.float32, device=self.device),
                                                   event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                                   rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
                u = torch.rand(1, dtype=torch.float32, device=self.device)
                tau_candidate = -torch.log(u) / current_intensity
                t_candidate = t_current + tau_candidate
                if t_candidate-prev_times > max_time:
                    next_times.append(torch.inf)
                    break
                candidate_intensity = self.intensity(given_times=torch.tensor([t_candidate], dtype=torch.float32, device=self.device),
                                                     event_times=event_times, event_types=event_types, event_meass=event_meass, 
                                                     rule_times=rule_times, rule_types=rule_types, rule_meass=rule_meass)
                if torch.rand(1, dtype=torch.float32, device=self.device) < (candidate_intensity / current_intensity):
                    next_times.append(t_candidate - prev_times)
                    break
                else:
                    t_current = t_candidate
                iteration += 1
            valid_samples = torch.tensor([t for t in next_times if t < torch.inf])
            # print(torch.mean(valid_samples) if len(valid_samples)!=0 else max_time)
            if len(valid_samples) == 0:
                return torch.tensor(max_time, dtype=torch.float32, device=self.device)
            return torch.mean(valid_samples)

# Custom Dataset
class EventDataset(Dataset):
    def __init__(self, data, rule_event_data):
        self.data = data
        self.rule_events_data = rule_event_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        events = self.data[idx]
        rule_events = self.rule_events_data[idx]
        return events, rule_events


# Custom collate function for variable-length sequences
def collate_fn(batch):
    events, rule_events = zip(*batch)
    return list(events), list(rule_events)


def train_epoch(model, optimizer, train_dataloader, device="cpu"):
    """
    Performs a single training epoch, where the model is updated based on the loss from the training data.

    Parameters:
    - model (nn.Module): The model being trained.
    - optimizer (torch.optim.Optimizer): The optimizer used for gradient descent.
    - train_dataloader (DataLoader): The data loader providing batches of training data.
    - device (str): The device ('cpu' or 'cuda') where the computation is performed.
    Returns:
    - total_loss (float): The accumulated loss for the epoch, averaged over all training batches.
    """
    train_size = sum(len(events) for events, _ in train_dataloader)
    model.train()
    total_loss = 0.0
    for events_batch, rule_events_batch in train_dataloader:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        for events, rule_events in zip(events_batch, rule_events_batch):
            event_times, event_types, event_meass = zip(*events)
            event_times = torch.tensor(event_times, dtype=torch.float32, device=device)
            event_types = torch.tensor(event_types, dtype=torch.long, device=device)
            event_meass = torch.tensor(event_meass, dtype=torch.float32, device=device)
            if rule_events == []:
                rule_times = torch.tensor([], dtype=torch.float32, device=device)
                rule_types = torch.tensor([], dtype=torch.long, device=device)
                rule_meass = torch.tensor([], dtype=torch.float32, device=device)
            else:
                rule_times, rule_types, rule_meass = zip(*rule_events)
                rule_times = torch.tensor(rule_times, dtype=torch.float32, device=device)
                rule_types = torch.tensor(rule_types, dtype=torch.long, device=device)
                rule_meass = torch.tensor(rule_meass, dtype=torch.float32, device=device)
            sequence_loss = model.forward(event_times, event_types, event_meass, rule_times, rule_types, rule_meass)
            batch_loss += sequence_loss / len(events_batch)
            total_loss += sequence_loss.item() / train_size
        batch_loss.backward()
        optimizer.step()
    return total_loss


def eval_epoch(model, eval_dataloader, target_name, device="cpu"):
    """
    Evaluates the model's performance on the validation or test dataset, computing NLL, MAE, and RMSE.

    Parameters:
    - model (nn.Module): The model being evaluated.
    - eval_dataloader (DataLoader): The data loader providing batches of evaluation data.
    - target_name (str): The name of the target variable for evaluation.
    - device (str): The device ('cpu' or 'cuda') where the computation is performed.
    Returns:
    - eval_nll (float): The average negative log-likelihood (NLL) on the evaluation dataset.
    - eval_mae (float): The average mean absolute error (MAE) on the evaluation dataset.
    - eval_rmse (float): The average root mean square error (RMSE) on the evaluation dataset.
    """
    model.eval()
    eval_size = sum(len(events) for events, _ in eval_dataloader)
    with torch.no_grad():
        eval_nll, eval_mae, eval_rmse = 0.0, 0.0, 0.0
        for events_batch, rule_events_batch in eval_dataloader:
            for events, rule_events in zip(events_batch, rule_events_batch):
                event_times, event_types, event_meass = zip(*events)
                event_times = torch.tensor(event_times, dtype=torch.float32, device=device)
                event_types = torch.tensor(event_types, dtype=torch.long, device=device)
                event_meass = torch.tensor(event_meass, dtype=torch.float32, device=device)
                if rule_events == []:
                    rule_times = torch.tensor([], dtype=torch.float32, device=device)
                    rule_types = torch.tensor([], dtype=torch.long, device=device)
                    rule_meass = torch.tensor([], dtype=torch.float32, device=device)
                else:
                    rule_times, rule_types, rule_meass = zip(*rule_events)
                    rule_times = torch.tensor(rule_times, dtype=torch.float32, device=device)
                    rule_types = torch.tensor(rule_types, dtype=torch.long, device=device)
                    rule_meass = torch.tensor(rule_meass, dtype=torch.float32, device=device)
                nll, mae, rmse = model.evaluate(event_times, event_types, event_meass, rule_times, rule_types, rule_meass, target_name)
                eval_nll += nll.item() / eval_size
                eval_mae += mae.item() / eval_size
                eval_rmse += rmse.item() / eval_size
    return eval_nll, eval_mae, eval_rmse


def train_model(model, data, rule_event_data, target_name, device="cpu", num_epochs=100, lr=0.01, patience=5, if_print=False):
    # Training settings
    train_prop = 0.8
    batch_size = 64
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_eval_loss = float('inf')
    patience_counter = 0
    # Training processes
    train_size = int(train_prop * len(data))
    test_size = len(data) - train_size
    train_size1 = int(0.25 * train_size)
    train_data, test_data = data[:train_size1], data[train_size:]
    train_rule_event_data, test_rule_event_data = rule_event_data[:train_size1], rule_event_data[train_size:]
    train_dataset = EventDataset(train_data, train_rule_event_data)
    test_dataset = EventDataset(test_data, test_rule_event_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    output_list = []
    for epoch in range(num_epochs):
        total_loss = train_epoch(model, optimizer, train_dataloader, device)
        eval_nll, eval_mae, eval_rmse = eval_epoch(model, test_dataloader, target_name, device)
        eval_loss = total_loss
        output_list.append([total_loss, eval_nll, eval_mae, eval_rmse])
        if if_print:
            print(f'Epoch {epoch}, Loss: {total_loss}')
            print(f'Eval NLL: {eval_nll}, Eval MAE: {eval_mae}, Eval RMSE: {eval_rmse}',)
        # Early stopping check
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if if_print:
                print("Early stopping triggered.")
            break
    return eval_loss, output_list


def generate_candidate_rules(variables, operators=["before","equal","and"], max_order=1):
    all_combinations = []
    for order in range(1, max_order + 1):
        for combination in product(variables, repeat=order):
            if len(set(combination)) == order:  # Ensure all variables are unique
                for ops in product(operators, repeat=order - 1):
                    rule = []
                    for i in range(order):
                        rule.append(combination[i])
                        if i < order - 1:
                            rule.append(ops[i])
                    all_combinations.append(" ".join(rule))
    return all_combinations


def score(data, var_name_dict, rules, target_name, device="cpu"):
    start_time = time.time()
    rule_set = RuleSet(data, var_name_dict)
    for rule in rules:
        rule_set.add_rule(rule)
    model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
    model.to(device)
    loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5)
    end_time = time.time()
    print(f"Score computation time: {end_time - start_time} seconds")
    return loss


class EarlyStopping:
    def __init__(self, tol=1e-4, patience=100):
        self.tol = tol
        self.patience = patience
        self.best_score = float('inf')
        self.counter = 0

    def __call__(self, res):
        if res.fun < self.best_score - self.tol:
            self.best_score = res.fun
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"Early stopping: No improvement for {self.patience} iterations.")
            return True
        return False

def generate_candidate_rules(variables, operators=["before","equal","and"], max_order=1, target_name=None, ref_rules=None):
    """
    Generates candidate rules based on variables, operators, and max order.
    """
    variables = [var for var in variables if var != target_name]
    all_combinations = []
    for order in range(1, max_order + 1):
        for combination in product(variables, repeat=order):
            if len(set(combination)) == order:  # Ensure unique variables
                for ops in product(operators, repeat=order - 1):
                    rule = []
                    for i in range(order):
                        rule.append(combination[i])
                        if i < order - 1:
                            rule.append(ops[i])
                    if ref_rules is None:
                        all_combinations.append(" ".join(rule))
                    else:
                        for ref_rule in ref_rules:
                            if ref_rule in rule:
                                all_combinations.append(" ".join(rule))
                                break
    return all_combinations


def evaluate_loss(temporary_rules, current_rules, data, var_name_dict, target_name, device="cpu"):
    """
    Evaluates the loss for a given set of rules.
    """
    rules = current_rules + temporary_rules
    loss = score(data, var_name_dict, rules, target_name, device)
    return float(loss)


def optimize(data, var_name_dict, target_name, max_order=1, num_candidates=10, n_calls=50, device="cpu"):
    """
    Optimizes the rule selection using Bayesian Optimization.
    """
    # Step 0: Find out the 1st order rules
    print("Optimizing rule selection...")
    selected_proportion = 1.0
    rule_set = RuleSet(data, var_name_dict)
    model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
    model.to(device)
    basic_loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5)
    variables = list(var_name_dict.keys())
    first_order_rules = generate_candidate_rules(variables, max_order=1, target_name=target_name)
    rule_losses = []
    for first_order_rule in tqdm(first_order_rules):
        rule_set = RuleSet(data, var_name_dict)
        rule_set.add_rule(first_order_rule)
        model = RuleBasedTPP(rule_set.var_name_dict, rule_set.rule_name_dict, rule_set.rule_var_ids, device=device)
        model.to(device)
        loss, _ = train_model(model, data, rule_set.rule_event_data, target_name, device, num_epochs=100, lr=0.01, patience=5)
        if loss < basic_loss:
            rule_losses.append((first_order_rule, loss))
    rule_losses.sort(key=lambda x: x[1])
    num_selected = int(selected_proportion * num_candidates)
    if len(rule_losses) >= num_selected:
        selected_first_rules = [rule for rule, _ in rule_losses[:num_selected]]
        print(f"Number of selected first order rules: {num_selected}")
    if len(rule_losses) == 0:
        print("No rule found.")
        return [], basic_loss
    
    # Step 1: Generate candidate rules
    candidate_rules = generate_candidate_rules(variables, max_order=max_order, target_name=target_name, ref_rules=selected_first_rules)
    print(f"Number of candidate rules: {len(candidate_rules)}")
    random.shuffle(candidate_rules)
    rule_indices = list(range(len(candidate_rules)))

    # Step 2: Define the search space
    search_space = [Categorical(rule_indices, name=f"rule_{i}") for i in range(num_candidates)]

    # Step 3: Define the objective function
    @use_named_args(search_space)
    def objective(**kwargs):
        # Decode rule indices to get the actual rules
        selected_indices = [kwargs[f"rule_{i}"] for i in range(num_candidates)]
        selected_rules = [candidate_rules[idx] for idx in selected_indices]
        return evaluate_loss(selected_rules, [], data, var_name_dict, target_name, device)

    # Step 4: Run Bayesian Optimization
    early_stopping = EarlyStopping(tol=1e-4, patience=100)
    result = forest_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,  # Number of iterations
        # n_initial_points=100,  # Number of random points
        n_jobs=-1,        # Number of cores to use
        random_state=24,
        callback=[early_stopping],  # Early stopping
        verbose=True
    )

    # Step 5: Extract the best rules and their loss
    best_indices = result.x  # Best rule indices
    best_rules = [candidate_rules[idx] for idx in best_indices]
    best_loss = result.fun

    return best_rules, best_loss
