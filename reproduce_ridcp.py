import time
import torch
import os
import sys

# Add path to import RIDCP
sys.path.append(os.path.join(os.getcwd(), 'restoration_tools'))

try:
    from agent_tools.RIDCP.inference_ridcp import ridcp_predict, load_ridcp_model
    
    img_path = '/home/LXJ/Python_Projects/AIA_Restore/原始数据/train/fog_series/fog/000013.png'
    device = 'cuda:0'
    
    print(f"Starting load_ridcp_model to {device}...")
    start_load = time.time()
    # Mocking or providing correct args for load_ridcp_model if we knew the signature
    # Based on common patterns in such repos. If it fails, we check code.
    model = load_ridcp_model(device=device)
    end_load = time.time()
    print(f"Load finished in {end_load - start_load:.2f}s")
    
    print(f"Starting ridcp_predict for {img_path}...")
    start_infer = time.time()
    result = ridcp_predict(model, img_path, device=device)
    end_infer = time.time()
    print(f"Inference finished in {end_infer - start_infer:.2f}s")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
