#!/usr/bin/env python3
"""
=============================================================
 AlgorimAI - Model Conversion Tools
 Convert AlgorimSeek to GGUF / ONNX / PyTorch / Transformers
=============================================================
"""

import os, sys, json, shutil, argparse, subprocess
from pathlib import Path

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     AlgorimSeek Model Converter v1.0                     ║
║     Supports: GGUF | ONNX | PyTorch | Transformers       ║
╚══════════════════════════════════════════════════════════╝
"""

# ── GGUF Conversion (llama.cpp) ──────────────────────────────
def convert_to_gguf(model_path: str, output_dir: str, quantization: str = "Q4_K_M"):
    """
    Convert HuggingFace model to GGUF format for llama.cpp / Ollama.
    
    quantization options:
        Q2_K   - Very small, low quality
        Q4_0   - Small, good quality
        Q4_K_M - Recommended balance (default)
        Q5_K_M - Good quality
        Q8_0   - High quality, larger
        F16    - Full precision (largest)
    """
    print(f"\n🔄 Converting to GGUF ({quantization})...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Install llama.cpp Python bindings
    print("📦 Checking llama.cpp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "-q"], check=True)
    
    # Step 2: Convert to F16 GGUF first
    f16_path = output_path / "algorimseek_f16.gguf"
    convert_script = "convert_hf_to_gguf.py"
    
    print(f"🔄 Converting {model_path} → F16 GGUF...")
    cmd = [
        sys.executable, convert_script,
        model_path,
        "--outtype", "f16",
        "--outfile", str(f16_path)
    ]
    # Note: download convert script from llama.cpp repo if needed
    # subprocess.run(cmd, check=True)
    
    # Step 3: Quantize
    q_path = output_path / f"algorimseek_{quantization}.gguf"
    print(f"🔄 Quantizing to {quantization}...")
    quant_cmd = ["./quantize", str(f16_path), str(q_path), quantization]
    # subprocess.run(quant_cmd, check=True)
    
    # Create Modelfile for Ollama
    modelfile = f"""FROM {q_path}

SYSTEM \"\"\"You are AlgorimSeek, an AI expert in the Algorim programming language (algo a47).
You generate, analyze, compile, and debug Algorim code.
Support commands: /imagine /compile /debug /execute /explain\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096
"""
    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(modelfile)
    
    # Create install script
    install_script = f"""#!/bin/bash
# Install AlgorimSeek in Ollama
ollama create algorimseek -f {modelfile_path}
echo "✅ AlgorimSeek installed in Ollama!"
echo "Run: ollama run algorimseek"
"""
    install_path = output_path / "install_ollama.sh"
    install_path.write_text(install_script)
    install_path.chmod(0o755)
    
    print(f"✅ GGUF conversion config ready in {output_dir}")
    print(f"   - Modelfile: {modelfile_path}")
    print(f"   - Install:   bash {install_path}")
    return str(q_path)

# ── ONNX Conversion ──────────────────────────────────────────
def convert_to_onnx(model_path: str, output_dir: str, optimize: bool = True):
    """
    Convert to ONNX format for inference acceleration.
    Compatible with ONNX Runtime, TensorRT, DirectML.
    """
    print(f"\n🔄 Converting to ONNX...")
    
    try:
        from optimum.exporters.onnx import main_export
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("📦 Installing optimum[onnxruntime]...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                        "optimum[onnxruntime-gpu]", "-q"], check=True)
        from optimum.exporters.onnx import main_export
    
    output_path = Path(output_dir) / "onnx"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Exporting {model_path} → ONNX...")
    main_export(
        model_name_or_path=model_path,
        output=str(output_path),
        task="text-generation-with-past",
        dtype="fp16",
        optimize="O2" if optimize else None,
        trust_remote_code=True
    )
    
    if optimize:
        print("⚡ Optimizing ONNX model...")
        # Post-export optimization
        ort_model = ORTModelForCausalLM.from_pretrained(
            str(output_path),
            provider="CUDAExecutionProvider"
        )
        ort_model.save_pretrained(str(output_path / "optimized"))
    
    print(f"✅ ONNX model saved to {output_path}")
    
    # Create inference example
    onnx_example = f"""
# ONNX Runtime Inference Example
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model = ORTModelForCausalLM.from_pretrained("{output_path}")
tokenizer = AutoTokenizer.from_pretrained("{model_path}")

prompt = "Write an Algorim action to sort an array"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
    (output_path / "inference_example.py").write_text(onnx_example)
    return str(output_path)

# ── PyTorch Export ───────────────────────────────────────────
def export_pytorch(model_path: str, output_dir: str):
    """
    Export to PyTorch format (TorchScript / pt files).
    """
    print(f"\n🔄 Exporting to PyTorch...")
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    output_path = Path(output_dir) / "pytorch"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    model.eval()
    
    # Save full model state dict
    state_dict_path = output_path / "algorimseek_weights.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config.to_dict(),
        'model_type': 'algorimseek',
        'version': '1.0'
    }, state_dict_path)
    
    # TorchScript export (for deployment without Python)
    try:
        print("🔄 Tracing to TorchScript...")
        dummy_input = tokenizer("test", return_tensors="pt")
        traced = torch.jit.trace(model, (dummy_input['input_ids'],))
        script_path = output_path / "algorimseek_scripted.pt"
        torch.jit.save(traced, str(script_path))
        print(f"✅ TorchScript saved: {script_path}")
    except Exception as e:
        print(f"⚠️  TorchScript failed (normal for complex models): {e}")
    
    print(f"✅ PyTorch weights saved: {state_dict_path}")
    return str(output_path)

# ── HuggingFace Transformers Export ─────────────────────────
def export_transformers(model_path: str, output_dir: str, 
                         quantize_bits: int = None, push_to_hub: str = None):
    """
    Export/merge LoRA adapters + save as standard HuggingFace model.
    Optionally quantize with GPTQ or push to Hub.
    """
    print(f"\n🔄 Exporting as Transformers model...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from peft import PeftModel, PeftConfig
    import torch
    
    output_path = Path(output_dir) / "transformers"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if this is a LoRA model
    peft_config_path = Path(model_path) / "adapter_config.json"
    
    if peft_config_path.exists():
        print("🔄 Detected LoRA adapter — merging with base model...")
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_id = peft_config.base_model_name_or_path
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("✅ LoRA merged")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Add AlgorimSeek model card
    model.config.model_type = "algorimseek"
    model.config._name_or_path = "AlgorimSeek"
    
    # Save in safetensors format
    print(f"💾 Saving to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output_path)
    
    # Create model card
    model_card = """---
language:
- ar
- en
license: apache-2.0
tags:
- algorim
- code
- vision
- arabic
- algorimseek
---

# AlgorimSeek

AI model specialized in the **Algorim programming language (algo a47)**.

## Capabilities
- Generate Algorim code from text/image
- Compile to bytecode (AlgorimVM)
- Debug and explain algorithms
- Bilingual (Arabic + English)
- Vision: read code screenshots

## Commands
- `/imagine` - Generate code visualization
- `/compile` - Compile to bytecode
- `/debug`   - Step-by-step execution
- `/execute` - Run code
- `/explain` - Detailed explanation

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("AlgorimSeek")
tokenizer = AutoTokenizer.from_pretrained("AlgorimSeek")

prompt = "Write an Algorim action to implement binary search"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
"""
    (output_path / "README.md").write_text(model_card)
    
    # Push to HuggingFace Hub
    if push_to_hub:
        print(f"🚀 Pushing to Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, safe_serialization=True)
        tokenizer.push_to_hub(push_to_hub)
        print(f"✅ Pushed to hub.huggingface.co/{push_to_hub}")
    
    print(f"✅ Transformers model saved: {output_path}")
    return str(output_path)

# ── Apple MLX Conversion ─────────────────────────────────────
def convert_to_mlx(model_path: str, output_dir: str):
    """Convert to Apple MLX format (for Apple Silicon)."""
    print(f"\n🍎 Converting to MLX (Apple Silicon)...")
    mlx_script = f"""
# Install: pip install mlx-lm
from mlx_lm import convert, load, generate

# Convert from HuggingFace
convert(
    hf_path="{model_path}",
    mlx_path="{output_dir}/mlx",
    quantize=True,
    q_bits=4
)

# Load and use
model, tokenizer = load("{output_dir}/mlx")
response = generate(model, tokenizer, 
    prompt="Write factorial in Algorim",
    max_tokens=256)
print(response)
"""
    mlx_path = Path(output_dir) / "convert_mlx.py"
    mlx_path.write_text(mlx_script)
    print(f"✅ MLX conversion script: {mlx_path}")
    print("   Run: python convert_mlx.py")

# ── Create Ollama Config ──────────────────────────────────────
def create_ollama_config(model_path: str, output_dir: str):
    """Create Ollama Modelfile for local deployment."""
    print(f"\n🦙 Creating Ollama configuration...")
    
    modelfile = """FROM ./algorimseek_Q4_K_M.gguf

SYSTEM \"\"\"You are AlgorimSeek — expert in Algorim programming language (algo a47).

Algorim syntax:
- Action name(params): type → begin...end
- Assignment: x <--- value
- For loop: for (i <--- 0 to n) do...endfor
- While: while (cond) do...done
- If: if (cond) then...else...endif
- Types: int, float, bool, char, arr, node

Commands: /imagine /compile /debug /execute /explain\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9  
PARAMETER num_ctx 8192
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
\"\"\"
"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "Modelfile").write_text(modelfile)
    
    run_script = """#!/bin/bash
# Create and run AlgorimSeek in Ollama
ollama create algorimseek -f Modelfile
echo ""
echo "✅ AlgorimSeek ready!"
echo ""
echo "Usage:"
echo "  ollama run algorimseek"
echo "  ollama run algorimseek 'Write bubble sort in Algorim'"
"""
    run_path = output_path / "run_ollama.sh"
    run_path.write_text(run_script)
    run_path.chmod(0o755)
    
    print(f"✅ Ollama config saved to {output_dir}")
    print("   Run: bash run_ollama.sh")

# ── Main CLI ─────────────────────────────────────────────────
def main():
    print(BANNER)
    
    parser = argparse.ArgumentParser(description="AlgorimSeek Model Converter")
    parser.add_argument("--model", required=True, help="Path to AlgorimSeek model")
    parser.add_argument("--output", default="./converted", help="Output directory")
    parser.add_argument("--format", nargs="+", 
                        choices=["gguf","onnx","pytorch","transformers","mlx","ollama","all"],
                        default=["transformers"],
                        help="Target format(s)")
    parser.add_argument("--quant", default="Q4_K_M", 
                        choices=["Q2_K","Q4_0","Q4_K_M","Q5_K_M","Q8_0","F16"],
                        help="GGUF quantization")
    parser.add_argument("--push-to-hub", default=None, help="HuggingFace Hub repo")
    
    args = parser.parse_args()
    
    formats = args.format
    if "all" in formats:
        formats = ["gguf","onnx","pytorch","transformers","mlx","ollama"]
    
    print(f"📁 Model: {args.model}")
    print(f"📁 Output: {args.output}")
    print(f"🎯 Formats: {formats}")
    print()
    
    results = {}
    
    if "transformers" in formats:
        results["transformers"] = export_transformers(
            args.model, args.output, push_to_hub=args.push_to_hub
        )
    
    if "pytorch" in formats:
        results["pytorch"] = export_pytorch(args.model, args.output)
    
    if "gguf" in formats:
        results["gguf"] = convert_to_gguf(args.model, args.output, args.quant)
    
    if "onnx" in formats:
        results["onnx"] = convert_to_onnx(args.model, args.output)
    
    if "mlx" in formats:
        convert_to_mlx(args.model, args.output)
        results["mlx"] = f"{args.output}/mlx"
    
    if "ollama" in formats:
        create_ollama_config(args.model, f"{args.output}/ollama")
        results["ollama"] = f"{args.output}/ollama"
    
    print("\n" + "═"*60)
    print("✅ CONVERSION SUMMARY")
    print("═"*60)
    for fmt, path in results.items():
        print(f"  {fmt.upper():15} → {path}")
    print("═"*60)

if __name__ == "__main__":
    main()
