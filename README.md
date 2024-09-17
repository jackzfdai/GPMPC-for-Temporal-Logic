# GPMPC-for-Temporal-Logic

Simulation code for the paper "Model Predictive Control for Systems with Partially Unknown Dynamics Under Signal Temporal Logic Specifications". 

The CSTR simulation can be executed by running "cstr_control.py". The autonomous vehicle double lane change simulation can be executed by running autonomous-car-control.py. 

Trajectory data from previous executions of both simulations are available in the trace_data folder, and can be parsed for the performance metrics and figures presented in the paper by running analyze_cstr.py and analyze_autocar.py. 

Note: A GUROBI license is required to run the double lane change case study, as well as the analyze_*.py scripts. A free academic license can be obtained from www.gurobi.com.