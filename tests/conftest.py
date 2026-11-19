import os
import sys

# Permite "from src..." ao rodar "python -m pytest" na raiz do projeto.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
