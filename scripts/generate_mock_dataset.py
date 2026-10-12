"""Generate only four fictional fixture CSVs; no original data input."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mock_dataset import write_mock,SEED
if __name__=='__main__':
    messages,people,locations,truth=write_mock()
    print(f'SYNTHETIC DEMO DATA: {len(messages)} messages, {len(people)} fictional participants, {len(locations)} locations; seed {SEED}. No transcript accessed.')
