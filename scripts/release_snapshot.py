"""Separate human-reviewed release operation; never called by processing or dashboard."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.storage_security import checked_path,verify_mode_isolation,restrict_permissions
from src.public_release import release_request,publish_fixed_snapshot

def main(argv=None):
    parser=argparse.ArgumentParser(description='Prepare private approval request or publish one fixed reviewed snapshot.')
    parser.add_argument('--action',choices=['prepare','publish'],required=True)
    parser.add_argument('--mode',choices=['synthetic','real'],required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--approve-public-release',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='publish' and not args.approve_public_release:
        parser.error('Explicit public release approval is required. No restricted input was read.')
    try:
        verify_mode_isolation(ROOT)
        source=checked_path(args.questions,ROOT)
        approval_path=checked_path(args.approval,ROOT)
        base=(ROOT/'data/private'/args.mode).resolve()
        if not source.is_relative_to(base) or not approval_path.is_relative_to(base) or source==approval_path:
            raise ValueError('Invalid scope')
        questions=pd.read_csv(source)
        if args.action=='prepare':
            with approval_path.open('x',encoding='utf-8') as stream:
                json.dump(release_request(questions,args.mode),stream,indent=2)
            restrict_permissions(approval_path)
        else:
            approval=json.loads(approval_path.read_text(encoding='utf-8'))
            publish_fixed_snapshot(ROOT,args.mode,questions,approval)
    except Exception:
        print('Release stopped: policy, approval, isolation, integrity or access validation failed. No source values logged.',file=sys.stderr)
        return 2
    print('Private approval request prepared.' if args.action=='prepare' else 'Fixed reviewed snapshot created; dashboard selection remains a separate approval step.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
