from pathlib import Path
from launch import configure
if __name__ == '__main__':
    default = Path.home()/'Downloads/BrainTumor_AI_Stage2/BrainTumor_AI_Stage2'
    project = input(f'Main trained project folder [{default}]: ').strip().strip('"') or str(default)
    classifier = input('Original BTC_IXI_Training results folder (optional, Enter to skip): ').strip().strip('"')
    configure(project, classifier or None)
