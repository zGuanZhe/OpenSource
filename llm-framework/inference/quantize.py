import torch

def quantize_model(model, dtype=torch.int8):
    """
    Dummy quantization function to demonstrate the concept.
    In a real scenario, use torch.quantization.quantize_dynamic 
    or custom bitsandbytes quantization.
    """
    print(f"Quantizing model to {dtype}...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=dtype
    )
    return quantized_model
