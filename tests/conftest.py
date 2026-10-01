import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get('COMFYUI_ROOT', ROOT.parent.parent))
sys.path.insert(0, str(COMFY))
import comfy.cli_args
comfy.cli_args.args.cpu = True
spec = importlib.util.spec_from_file_location('h3_continuity', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules['h3_continuity'] = package
spec.loader.exec_module(package)
sys.modules.setdefault('__init__', package)
