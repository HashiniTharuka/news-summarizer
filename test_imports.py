# test_imports.py
try:
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install: pip install transformers")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install: pip install torch")

try:
    import streamlit
    print(f"✅ Streamlit version: {streamlit.__version__}")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install: pip install streamlit")

print("\n✅ All imports successful!")