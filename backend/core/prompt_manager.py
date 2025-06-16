import sys
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class PromptTemplateManager:
    def __init__(self, template_path):
        self.template_path = template_path
        self.templates = self._load_templates()

    def _load_templates(self):
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading templates from {self.template_path}: {e}")
            return {}

    def list_templates(self):
        # Returns all keys and titles for UI dropdowns etc.
        return [{"key": k, "title": v.get("title", k)} for k, v in self.templates.items()]

    def get_template(self, key):
        return self.templates.get(key)
