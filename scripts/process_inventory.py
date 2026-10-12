"""Explicit local CLI. Do not execute real mode without the user's approval."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.local_processing import process_inventory, synthetic_inventory, PROJECT_ROOT
from src.restricted_inventory import InventoryValidationError
from src.embedding_model import ModelUnavailableError

def main(argv=None):
    parser = argparse.ArgumentParser(description='Local restricted inventory analysis; no dashboard publication.')
    parser.add_argument('--mode', required=True, choices=['synthetic','real'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--approve-real-processing', action='store_true')
    parser.add_argument('--assume-timezone', default=None)
    parser.add_argument('--deduplicate-exact', action='store_true')
    parser.add_argument('--minimum-participants', type=int, default=5)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models' / 'all-MiniLM-L6-v2')
    args = parser.parse_args(argv)
    if args.mode == 'real' and not args.approve_real_processing:
        parser.error('Real processing requires explicit user approval and --approve-real-processing. No CSV has been read.')
    if args.mode == 'real' and args.input is None:
        parser.error('Real mode requires --input inside data/private/ or data/staging/.')
    if args.mode == 'synthetic' and (args.input is not None or args.approve_real_processing):
        parser.error('Synthetic mode generates its own fixture; --input and real approval are forbidden.')
    try:
        from src.local_processing import run_paths, restricted_path
        from src.storage_security import secure_directory, restrict_permissions
        run_paths(ROOT,args.mode,args.run_id)
        path = args.input
        if args.mode == 'synthetic':
            path = ROOT / 'data' / 'staging' / 'synthetic' / 'fixtures' / f'{args.run_id}.csv'
            path = restricted_path(path,ROOT)
            if path.exists():
                raise InventoryValidationError('Synthetic fixture already exists; choose another run ID.')
            secure_directory(path.parent,ROOT)
            synthetic_inventory().to_csv(path,index=False)
            restrict_permissions(path)
        process_inventory(path, mode=args.mode,run_id=args.run_id,authorized=args.approve_real_processing,
                          assume_timezone=args.assume_timezone,deduplicate_exact=args.deduplicate_exact,
                          minimum_participants=args.minimum_participants,model_dir=args.model_dir)
    except (InventoryValidationError,ModelUnavailableError) as error:
        print('Processing stopped: validation, approval, model availability, or storage security failed. No source values are logged.',file=sys.stderr)
        return 2
    except Exception:
        # Never emit a traceback that could contain message-level values.
        print('Processing stopped due to a local processing error. No source values were logged.',file=sys.stderr)
        return 2
    print(f'{args.mode.capitalize()} processing complete. Restricted review records and draft aggregates saved locally; nothing published.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
